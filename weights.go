package main

// Weights live in Hanzo S3, never in git and never in the image. A pod fetches
// them at boot into its own volume; a laptop or a dev box keeps one copy in the
// models directory and fetches nothing it already has.
//
// weights.sum is the whole contract: one line per file — sha256, size, path —
// and it is compiled into the binary, so a build names exactly the bytes it was
// tested against. A download is written beside its path, hashed as it streams,
// and renamed into place only when the hash agrees; a file that is already there
// at the right size is a file a previous boot verified (or one a person put
// there on purpose) and is kept.

import (
	"bufio"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed weights.sum
var weightsSum string

// prefix is where the weights sit in the bucket: speech/<path in weights.sum>.
const prefix = "speech/"

type weight struct {
	sum  string
	size int64
	path string
}

func manifest() ([]weight, error) {
	var out []weight
	sc := bufio.NewScanner(strings.NewReader(weightsSum))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		f := strings.SplitN(line, " ", 3)
		if len(f) != 3 || len(f[0]) != 64 {
			return nil, fmt.Errorf("weights.sum: malformed line %q", line)
		}
		n, err := strconv.ParseInt(f[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("weights.sum: size in %q: %w", line, err)
		}
		out = append(out, weight{sum: f[0], size: n, path: f[2]})
	}
	return out, sc.Err()
}

// store is the S3 location weights are fetched from. Its zero value fetches
// nothing, which is right for a machine that already holds them.
type store struct {
	endpoint, bucket, region, key, secret string
}

func storeFromEnv() store {
	return store{
		endpoint: strings.TrimRight(os.Getenv("S3_ENDPOINT"), "/"),
		bucket:   os.Getenv("S3_BUCKET"),
		region:   envOr("S3_REGION", "us-east-1"),
		key:      os.Getenv("AWS_ACCESS_KEY_ID"),
		secret:   os.Getenv("AWS_SECRET_ACCESS_KEY"),
	}
}

// fetch makes dir hold every file weights.sum names.
func fetch(ctx context.Context, dir string, s store) error {
	all, err := manifest()
	if err != nil {
		return err
	}
	var missing []weight
	for _, w := range all {
		if fi, err := os.Stat(filepath.Join(dir, w.path)); err == nil && fi.Size() == w.size {
			continue
		}
		missing = append(missing, w)
	}
	if len(missing) == 0 {
		return nil
	}
	if s.endpoint == "" || s.bucket == "" {
		return fmt.Errorf("%d weight files missing under %s (first: %s) and no S3_ENDPOINT/S3_BUCKET to fetch them from", len(missing), dir, missing[0].path)
	}
	slog.Info("fetching weights", "files", len(missing), "from", s.endpoint+"/"+s.bucket+"/"+prefix)
	began := time.Now()
	jobs := make(chan weight)
	errs := make(chan error, len(missing))
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for w := range jobs {
				if err := s.get(ctx, dir, w); err != nil {
					errs <- err
				}
			}
		}()
	}
	for _, w := range missing {
		jobs <- w
	}
	close(jobs)
	wg.Wait()
	close(errs)
	if err := <-errs; err != nil {
		return err
	}
	slog.Info("weights fetched", "files", len(missing), "seconds", time.Since(began).Seconds())
	return nil
}

func (s store) get(ctx context.Context, dir string, w weight) error {
	dst := filepath.Join(dir, w.path)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	var last error
	for attempt := range 4 {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
		}
		if last = s.download(ctx, dst, w); last == nil {
			return nil
		}
		slog.Warn("weight fetch failed", "path", w.path, "attempt", attempt+1, "err", last)
	}
	return fmt.Errorf("fetch %s: %w", w.path, last)
}

// fetcher answers a GET's headers within headerWait or the attempt is abandoned
// and retried; a body that stops arriving for stall is abandoned the same way.
// A connection that hangs is the failure that otherwise never ends: the pod sits
// unready forever with nothing in its log.
var fetcher = &http.Client{Transport: &http.Transport{
	Proxy:                 http.ProxyFromEnvironment,
	ResponseHeaderTimeout: headerWait,
	MaxIdleConnsPerHost:   8,
}}

const (
	headerWait = 30 * time.Second
	stall      = 60 * time.Second
)

func (s store) download(ctx context.Context, dst string, w weight) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	idle := time.AfterFunc(stall, cancel)
	defer idle.Stop()
	req, err := s.request(ctx, prefix+w.path)
	if err != nil {
		return err
	}
	resp, err := fetcher.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return fmt.Errorf("s3 answered %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	part := dst + ".part"
	f, err := os.Create(part)
	if err != nil {
		return err
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, h), watched{resp.Body, idle})
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(part)
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); n != w.size || got != w.sum {
		os.Remove(part)
		return fmt.Errorf("got %d bytes hashing %s; weights.sum says %d bytes, %s", n, got, w.size, w.sum)
	}
	return os.Rename(part, dst)
}

// watched resets the stall timer every time bytes arrive.
type watched struct {
	r     io.Reader
	timer *time.Timer
}

func (w watched) Read(p []byte) (int, error) {
	n, err := w.r.Read(p)
	if n > 0 {
		w.timer.Reset(stall)
	}
	return n, err
}

// request is a GET for one object, signed with AWS Signature Version 4,
// path-style (hanzoai/s3 serves the bucket in the path).
func (s store) request(ctx context.Context, key string) (*http.Request, error) {
	u, err := url.Parse(s.endpoint)
	if err != nil {
		return nil, err
	}
	path := "/" + s.bucket + "/" + escape(key)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.endpoint+path, nil)
	if err != nil {
		return nil, err
	}
	req.URL.RawPath = path
	now := time.Now().UTC()
	stamp, day := now.Format("20060102T150405Z"), now.Format("20060102")
	const payload = "UNSIGNED-PAYLOAD"
	req.Header.Set("x-amz-date", stamp)
	req.Header.Set("x-amz-content-sha256", payload)
	if s.key == "" {
		return req, nil // an anonymous read, for a bucket that allows one
	}
	canonical := strings.Join([]string{
		http.MethodGet, path, "",
		"host:" + u.Host, "x-amz-content-sha256:" + payload, "x-amz-date:" + stamp, "",
		"host;x-amz-content-sha256;x-amz-date", payload,
	}, "\n")
	scope := day + "/" + s.region + "/s3/aws4_request"
	sum := sha256.Sum256([]byte(canonical))
	toSign := "AWS4-HMAC-SHA256\n" + stamp + "\n" + scope + "\n" + hex.EncodeToString(sum[:])
	k := mac([]byte("AWS4"+s.secret), day)
	k = mac(k, s.region)
	k = mac(k, "s3")
	k = mac(k, "aws4_request")
	req.Header.Set("Authorization", fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=%s",
		s.key, scope, hex.EncodeToString(mac(k, toSign))))
	return req, nil
}

func mac(key []byte, msg string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(msg))
	return h.Sum(nil)
}

// escape is SigV4's URI encoding: every byte but A-Z a-z 0-9 - . _ ~ is
// percent-encoded, and the slashes between segments stay.
func escape(key string) string {
	var b strings.Builder
	for i := 0; i < len(key); i++ {
		c := key[i]
		switch {
		case 'A' <= c && c <= 'Z', 'a' <= c && c <= 'z', '0' <= c && c <= '9', c == '-', c == '.', c == '_', c == '~', c == '/':
			b.WriteByte(c)
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

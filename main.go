// Command speech is Hanzo's own CPU speech: parakeet and whisper hear, kokoro
// speaks, on the OpenAI audio shape the ai plane already routes to.
//
// It is a service of its own rather than code inside the cloud binary because
// sherpa-onnx is C++ reached through cgo, and cloud builds with CGO_ENABLED=0.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	port := envOr("PORT", "8000")
	s := &server{here: here(port), maxIn: 100 << 20}

	srv := &http.Server{Addr: ":" + port, Handler: s.routes(), ReadHeaderTimeout: 10 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	go func() {
		dir := envOr("MODELS", "/models")
		began := time.Now()
		if err := fetch(ctx, dir, storeFromEnv()); err != nil {
			slog.Error("weights unavailable", "err", err)
			os.Exit(1)
		}
		sp, err := load(dir)
		if err != nil {
			slog.Error("models failed to load", "err", err)
			os.Exit(1)
		}
		s.sp.Store(sp)
		slog.Info("ready", "models", sp.models(), "seconds", time.Since(began).Seconds())
	}()

	go func() {
		<-ctx.Done()
		// Drain: stop taking connections, let requests in flight finish. A
		// growing transcript lives in this process and goes with it, as it always
		// has — the window was only ever in this pod's memory.
		shut, cancel := context.WithTimeout(context.Background(), 25*time.Second)
		defer cancel()
		_ = srv.Shutdown(shut)
	}()
	slog.Info("listening", "port", port, "at", s.here)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("server", "err", err)
		os.Exit(1)
	}
}

// here is where THIS process answers. A growing transcript is a window in one
// process's memory, and a Service address round-robins per connection, so
// `open` says which pod took the session and the caller addresses that pod for
// the rest of it. Empty without POD_IP, which is the honest answer for a single
// process: there is nothing to pin to.
func here(port string) string {
	ip := os.Getenv("POD_IP")
	if ip == "" {
		return ""
	}
	return fmt.Sprintf("http://%s:%s", ip, port)
}

// load opens every model, side by side: they are independent, and the slowest
// one sets the time to ready either way.
func load(dir string) (*speech, error) {
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		errs []error
	)
	sp := &speech{ears: map[string]ear{}, mouths: map[string]mouth{}}
	step := func(name string, f func() error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			began := time.Now()
			if err := f(); err != nil {
				mu.Lock()
				errs = append(errs, fmt.Errorf("%s: %w", name, err))
				mu.Unlock()
				return
			}
			slog.Info("loaded", "model", name, "seconds", time.Since(began).Seconds())
		}()
	}
	set := func(f func()) { mu.Lock(); f(); mu.Unlock() }
	step("parakeet", func() error {
		r, err := newParakeet(dir)
		if err != nil {
			return err
		}
		set(func() { sp.ears["parakeet"] = r })
		return nil
	})
	step("whisper", func() error {
		r, err := newWhisper(dir)
		if err != nil {
			return err
		}
		set(func() { sp.ears["whisper"] = r })
		return nil
	})
	step("lid", func() error {
		l, err := newIdentifier(dir)
		if err != nil {
			return err
		}
		set(func() { sp.lid = l })
		return nil
	})
	step("kokoro", func() error {
		k, err := newKokoro(dir)
		if err != nil {
			return err
		}
		set(func() { sp.mouths["kokoro"] = k })
		return nil
	})
	step("vad", func() error {
		v, err := newSilero(dir)
		if err != nil {
			return err
		}
		set(func() { sp.vad = v })
		return nil
	})
	wg.Wait()
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return sp, nil
}

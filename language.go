package main

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// parakeetLanguages are the 25 European languages Parakeet TDT 0.6B v3 was
// trained on. It identifies which of them it hears by itself; anything outside
// this set is whisper's.
var parakeetLanguages = set("bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
	"lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk")

// whisperLanguages are the codes whisper large-v3-turbo carries a token for —
// all_language_codes in its own metadata. A code outside this set is refused
// here, by name: handed to sherpa it ends the process (whisper decoder,
// "Invalid language" -> exit(-1)).
var whisperLanguages = set("en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
	"sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
	"hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
	"br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so",
	"af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa",
	"lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue")

func set(codes ...string) map[string]bool {
	m := make(map[string]bool, len(codes))
	for _, c := range codes {
		m[c] = true
	}
	return m
}

// language reads a caller's `language` as the code the models know.
//
// OpenAI's contract is ISO-639-1, and callers also send a region ("en-US",
// "pt_BR") or a capital; all of those name the same language. ISO's "jv" is
// whisper's "jw". Empty means the caller named none.
func language(raw string) (string, error) {
	code := strings.ToLower(strings.TrimSpace(raw))
	if i := strings.IndexAny(code, "-_"); i > 0 {
		code = code[:i]
	}
	switch code {
	case "":
		return "", nil
	case "jv":
		code = "jw"
	}
	if !whisperLanguages[code] {
		return "", fmt.Errorf("unknown language %q; send an ISO-639-1 code such as en, de, zh or ja", raw)
	}
	return code, nil
}

// kokoroLanguages are the languages a Kokoro v1.0 voice speaks, one per voice
// prefix (kokoroLang): American and British English, Spanish, French, Hindi,
// Italian, Japanese, Brazilian Portuguese and Mandarin.
var kokoroLanguages = set("en", "es", "fr", "hi", "it", "ja", "pt", "zh")

// supertonicLanguages are Supertonic 3's 31, from its model card. Arabic is the
// one this service carries it for; the rest come with it.
var supertonicLanguages = set("en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi",
	"hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi")

// script names the language a text's own letters settle: "ar" when most of its
// letters are Arabic script, "" otherwise. Only Arabic is read off the page —
// it is the one script a caller can send kokoro that none of its voices can say.
// Arabic script also writes Persian and Urdu, which supertonic does not speak;
// a caller with those names its language and is told so.
func script(text string) string {
	var letters, arabic int
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		letters++
		if unicode.Is(unicode.Arabic, r) {
			arabic++
		}
	}
	if letters > 0 && 2*arabic > letters {
		return "ar"
	}
	return ""
}

func sorted(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Breath is the quiet put between two sentences a stream synthesised apart.
const Breath = 0.15

// sentencesOf cuts text after each sentence's closing mark — Latin or Arabic —
// and at line breaks, keeping the mark with its sentence. A stream synthesises
// these one at a time, so the first can play while the next is made.
func sentencesOf(text string) []string {
	var out []string
	start := 0
	rs := []rune(text)
	for i, r := range rs {
		end := r == '\n' || ((r == '.' || r == '!' || r == '?' || r == '؟' || r == '؛' || r == '…') && (i+1 == len(rs) || unicode.IsSpace(rs[i+1])))
		if end {
			if s := strings.TrimSpace(string(rs[start : i+1])); s != "" {
				out = append(out, s)
			}
			start = i + 1
		}
	}
	if s := strings.TrimSpace(string(rs[start:])); s != "" {
		out = append(out, s)
	}
	return out
}

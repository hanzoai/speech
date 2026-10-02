package main

import (
	"fmt"
	"strings"
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

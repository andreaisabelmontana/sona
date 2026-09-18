# Dataset provenance

`SpotifyFeatures.csv` is an unchanged copy of the CSV included in [cayetana-h/Moodify](https://github.com/cayetana-h/Moodify).

- Source revision: [`f0596bbf19567b2400924b3c172dfb806eabae02`](https://github.com/cayetana-h/Moodify/tree/f0596bbf19567b2400924b3c172dfb806eabae02).
- Source file: [`SpotifyFeatures.csv`](https://github.com/cayetana-h/Moodify/blob/f0596bbf19567b2400924b3c172dfb806eabae02/SpotifyFeatures.csv).
- Copied for this recreation on 2026-09-18.
- SHA-256: `2628a9a70b4f108bae9e13420654011b02938110d7edd75b4f21832a65e0064b` (canonical bytes from the upstream Git blob, with LF line endings).
- Format: comma-separated UTF-8 text with a byte-order mark, a header, and 232,725 data rows.
- Distinct track IDs: 176,774. The additional 55,951 rows repeat track IDs, often under different genre listings.
- Comedy rows: 9,681. Excluding them leaves 223,044 rows and 167,101 distinct track IDs before any other cleaning.

The upstream README describes the audio features as originating from the Spotify Web API. The repository does not provide a separate dataset license, collection script, collection date, or complete provenance record. Its Apache-2.0 project license is retained for the code; this note does not assert that the code license grants rights to Spotify data or to the artists' content. Spotify and artist/track names belong to their respective owners. No audio recordings are included.

## Columns and conventions

The file contains track/artist names, track IDs, genre, popularity, duration, and Spotify-style audio descriptors. Sona uses `valence`, `energy`, and `acousticness` to create training pseudo-labels, then predicts them from other numeric audio descriptors and categorical musical features.

`key` contains note names (`A`, `A#`, `B`, `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`). `mode` contains `Major` or `Minor`. `time_signature` contains `0/4`, `1/4`, `3/4`, `4/4`, or `5/4`. Metadata is retained for display, but track IDs, names, artists, genre, and popularity are not classifier inputs.

Training filters and deduplicates in memory; it does not rewrite this source CSV. See [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for the evaluation procedure and interpretation limits.

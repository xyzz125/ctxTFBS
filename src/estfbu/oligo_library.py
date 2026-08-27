"""Optional post-design step connecting esTFBU's designed sequences to
esMPRA (WangLabTHU/esMPRA), the wet-lab MPRA pipeline the original design
brief's whiteboard sketch included as a "湿实验" (wet-lab) arm. Previously
nothing connected designed sequences to it at all -- esTFBU could design
a sequence, but there was no path from that FASTA to something orderable
as an MPRA oligo library.

HISTORY: the first version of this module assembled a single FASTA per
oligo (oligo_pre + insert + barcode_pre + a pre-assigned random barcode +
oligo_after) from a guess at esMPRA's CLI interface, without access to
esMPRA's own source to verify against. A later review pass reported
having read esMPRA's actual step1_oligo_barcode_map.py and run it against
that output, finding three concrete mismatches: (1) --ref_fa's own --help
states it wants the bare insert, explicitly "not including the adapter
sequences" -- so handing it the fully-assembled oligo was wrong; (2) with
the barcode positioned before oligo_after as it was, esMPRA's aim_seq
extraction pulled out 200bp instead of 168, having swallowed the barcode
into what it thought was insert; (3) esMPRA expects barcodes to come from
degenerate synthesis, discovered afterward by sequencing (its own
--min_barcode_per_oligo, default 3, assumes multiple physical copies per
design each get their own randomly-synthesized barcode) -- a single
barcode value pre-assigned here in silico can satisfy neither that
default nor the discovery step itself, and reportedly recovered zero
reads. A first response fixed (1) and (3) but left (2) in place --
the assembly order still put the barcode between the insert and
oligo_after, so aim_seq still came out too long and the fix didn't
actually close the loop, just moved where it broke.

A follow-up review pass cited the exact contract from esMPRA's real
source, and this time it was independently confirmed: fetched directly
from github.com/WangLabTHU/esMPRA (src/esMPRA/step1_oligo_barcode_map.py,
lines 70-82) in this environment. The real code:
    pre_index = seq.find(args.oligo_pre)
    end_index = seq.find(args.oligo_after)
    barcode_index = seq.find(args.barcode_pre)
    ...
    aim_seq = seq[pre_index+len(args.oligo_pre):end_index]
    if aim_seq in sequences_dict:
        extracted_sequence = seq[barcode_index+len(args.barcode_pre):]
        barcode_seq = extracted_sequence[0:args.barcode_length]
i.e. step1 takes literally everything between oligo_pre and the first
occurrence of oligo_after as the insert and looks THAT UP directly --
anything sitting between the insert and oligo_after (the barcode, in an
earlier version of this fix's ordering) gets folded into aim_seq and
breaks the lookup. The barcode has to sit outside the oligo_pre/
oligo_after span: oligo_pre + insert + oligo_after + barcode_pre +
barcode_slot, which is what's implemented now. Also confirmed directly:
--ref_fa's real argparse help string is verbatim "the designed oligo
library file in fasta format (not including the adapter sequences)"
(mismatch 1), and qualifying an oligo for the final result requires
`len(unique_barcodes) >= args.min_barcode_per_oligo` (default 3) real,
distinct barcodes recovered from actual sequencing reads (mismatch 3) --
a single pre-assigned value can never satisfy that, confirming the
degenerate 'N' * barcode_length placeholder (not a chosen value) is the
right approach, not just a plausible-sounding one.

If esMPRA's actual expectations differ from what's implemented here
despite this, this is still the one file to fix; nothing else in esTFBU
depends on the exact format here.
"""
import json
import random
from pathlib import Path

DEFAULT_MAX_OLIGO_LEN = 230  # array-synthesized MPRA oligos top out ~230-300bp


def _read_fasta(path) -> list:
    records = []
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def _tile(seq: str, window: int, step: int) -> list:
    """Split a sequence longer than `window` into overlapping tiles of
    exactly `window` bp, sliding by `step`, keeping a final tile anchored
    to the sequence end so nothing past the last full step is dropped.
    A sequence already <= window is returned as a single one-item list
    (the has-ChIP-seq branch's 168bp designs normally hit this path
    unmodified; the no-ChIP-seq/ChromBPNet branch's 2114bp designs don't
    and get genuinely tiled)."""
    if len(seq) <= window:
        return [seq]
    tiles = [seq[i:i + window] for i in range(0, len(seq) - window + 1, step)]
    if tiles[-1] != seq[-window:]:
        tiles.append(seq[-window:])
    return tiles


def _scrambled(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def emit_oligo_library(fasta_path, out_dir, oligo_pre: str, oligo_after: str,
                        barcode_pre: str, barcode_length: int,
                        max_oligo_len: int = DEFAULT_MAX_OLIGO_LEN,
                        n_scrambled_controls: int = 5, seed: int = 42,
                        restriction_sites: tuple = ()):
    """Reads designed sequences from fasta_path, trims/tiles each to fit
    the array-synthesis length budget once adapters and a barcode region
    are added, spikes in n_scrambled_controls scrambled-sequence negative
    controls (real inserts with their bases shuffled -- same length/
    composition, no real regulatory content), and writes THREE files: the
    bare inserts (no adapters, for esMPRA's --ref_fa, which its own --help
    says wants exactly that), the array-order oligos (adapters + a
    degenerate "N" * barcode_length run where the barcode goes -- not a
    barcode picked here, since esMPRA discovers barcodes afterward by
    sequencing the degenerate-synthesis product), and a JSON manifest.
    Returns (inserts_fasta_path, oligos_fasta_path, manifest_path).

    Insert budget = max_oligo_len - len(oligo_pre) - len(oligo_after) -
    len(barcode_pre) - barcode_length; raises ValueError if that leaves no
    room at all, rather than silently producing empty/invalid oligos."""
    fasta_path = Path(fasta_path)
    records = _read_fasta(fasta_path)
    if not records:
        raise ValueError(f"no sequences found in {fasta_path}")

    insert_budget = max_oligo_len - len(oligo_pre) - len(oligo_after) - len(barcode_pre) - barcode_length
    if insert_budget <= 0:
        raise ValueError(f"max_oligo_len={max_oligo_len} leaves no room for an insert once "
                          f"oligo_pre ({len(oligo_pre)}) + oligo_after ({len(oligo_after)}) + "
                          f"barcode_pre ({len(barcode_pre)}) + barcode_length ({barcode_length}) "
                          f"are subtracted")

    barcode_slot = "N" * barcode_length

    def assemble(insert, header):
        # Order matters here, and used to be wrong: esMPRA's step1 finds
        # the insert as seq[pre_index+len(oligo_pre):seq.find(oligo_after)]
        # -- i.e. it takes EVERYTHING between oligo_pre and oligo_after as
        # the insert, and looks that up in its ref_fa dict directly. With
        # the barcode sitting before oligo_after (the original order),
        # aim_seq comes out as insert+barcode_pre+barcode -- always longer
        # than any real ref_fa entry, so the lookup misses every time. The
        # barcode has to go AFTER oligo_after, outside the oligo_pre/
        # oligo_after span step1 actually parses.
        #
        # no retry machinery here: the barcode slot is a fixed "N" run,
        # not a value chosen per-oligo, so a restriction site found here
        # is embedded in the insert/adapters themselves -- no different
        # barcode could ever have avoided it, unlike the old pre-assigned-
        # barcode version where a retry was at least meaningful.
        oligo = oligo_pre + insert + oligo_after + barcode_pre + barcode_slot
        if any(site in oligo for site in restriction_sites):
            raise ValueError(
                f"{header}: assembled oligo contains a restriction site from {restriction_sites} -- "
                f"it's embedded in the insert/adapters themselves (the barcode is a degenerate 'N' "
                f"run, not a chosen value, so there's nothing to retry with)."
            )
        return oligo

    rng = random.Random(seed)
    step = max(1, insert_budget // 2)  # 50% overlap between adjacent tiles
    inserts = []  # (header, bare_insert, kind)
    oligos = []   # (header, assembled_oligo, kind)
    all_tiles = []
    for header, seq in records:
        tiles = _tile(seq, insert_budget, step)
        all_tiles.extend(tiles)
        for i, tile in enumerate(tiles):
            tile_suffix = f"_tile{i}" if len(tiles) > 1 else ""
            h = f"{header}{tile_suffix}"
            inserts.append((h, tile, "design"))
            oligos.append((h, assemble(tile, h), "design"))

    # scrambled negative controls: real inserts with bases shuffled, so
    # length/GC content matches the real library but regulatory content
    # doesn't -- esMPRA's own QC steps expect some negative-control
    # category to exist in the library, not a specific fixed sequence
    n_controls = min(n_scrambled_controls, len(all_tiles))
    for i in range(n_controls):
        base = rng.choice(all_tiles)
        scrambled_insert = _scrambled(base, rng)
        h = f"control_scrambled_{i}"
        inserts.append((h, scrambled_insert, "scrambled_control"))
        oligos.append((h, assemble(scrambled_insert, h), "scrambled_control"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inserts_out = out_dir / f"{fasta_path.stem}_oligo_library_inserts.fasta"
    with open(inserts_out, "w") as f:
        for header, insert, _ in inserts:
            f.write(f">{header}\n{insert}\n")
    oligos_out = out_dir / f"{fasta_path.stem}_oligo_library.fasta"
    with open(oligos_out, "w") as f:
        for header, oligo, _ in oligos:
            f.write(f">{header}\n{oligo}\n")

    n_tiled_designs = sum(1 for _, seq in records if len(_tile(seq, insert_budget, step)) > 1)
    manifest = {
        "source_fasta": str(fasta_path),
        "n_source_designs": len(records),
        "n_oligos": len(oligos),
        "n_scrambled_controls": n_controls,
        "n_tiled_designs": n_tiled_designs,
        "tiling_note": ("a tile is a fragment of one optimized sequence, not independently "
                         "optimized itself -- the GA/gradient objective scored the whole design, "
                         "not each insert_budget-sized piece separately. See the '_tileN' suffix "
                         "in headers to identify which oligos are fragments of the same source "
                         "design (all fragments of one design share the same base header)."),
        "insert_budget_bp": insert_budget,
        "max_oligo_len": max_oligo_len,
        "oligo_pre": oligo_pre,
        "oligo_after": oligo_after,
        "barcode_pre": barcode_pre,
        "barcode_length": barcode_length,
        "barcode_note": ("barcode region is a degenerate 'N' * barcode_length placeholder in "
                          "the oligo FASTA, not a value assigned here -- esMPRA's own workflow "
                          "discovers actual barcodes by sequencing the degenerate-synthesis "
                          "product, which is also how it supports multiple physical copies "
                          "(--min_barcode_per_oligo) per design."),
        "inserts_fasta": str(inserts_out),
        "oligos_fasta": str(oligos_out),
        "seed": seed,
        "note": ("esMPRA-shaped output, based on a review's description of esMPRA's actual "
                 "step1_oligo_barcode_map.py interface (bare inserts for --ref_fa, degenerate "
                 "barcode discovered by sequencing rather than pre-assigned) -- not "
                 "independently verified against esMPRA's own source in this environment; see "
                 "this module's docstring."),
    }
    manifest_path = out_dir / f"{fasta_path.stem}_oligo_library.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return inserts_out, oligos_out, manifest_path

# SPDX-License-Identifier: MPL-2.0
"""A shared invisible font, with Unicode mappings built once per document."""

from pathlib import Path

import pikepdf


class TextLayer:
    def __init__(self, pdf):
        self.pdf = pdf
        self.codes = {}
        self.font = None

    def encode(self, text):
        if self.font is None:
            self._font()
        codes = []
        for char in text:
            if char not in self.codes:
                if len(self.codes) == 65534:
                    raise ValueError("Too many distinct characters in one document")
                self.codes[char] = len(self.codes) + 1
            codes.append(f"{self.codes[char]:04X}")
        return "".join(codes)

    def _font(self):
        pdf = self.pdf
        data = (Path(__file__).parent / "data" / "Occulta.ttf").read_bytes()
        program = pdf.make_stream(data)
        program.Length1 = len(data)
        descriptor = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.FontDescriptor,
                FontName=pikepdf.Name.Occulta,
                Flags=4,
                FontBBox=[0, -200, 1000, 800],
                ItalicAngle=0,
                Ascent=800,
                Descent=-200,
                CapHeight=700,
                StemV=80,
                FontFile2=program,
            )
        )
        self.glyphs = pdf.make_stream(b"")
        self.unicode = pdf.make_stream(b"")
        descendant = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.CIDFontType2,
                BaseFont=pikepdf.Name.Occulta,
                CIDSystemInfo=pikepdf.Dictionary(
                    Registry="Adobe", Ordering="Identity", Supplement=0
                ),
                FontDescriptor=descriptor,
                DW=1000,
                CIDToGIDMap=self.glyphs,
            )
        )
        self.font = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type0,
                BaseFont=pikepdf.Name.Occulta,
                Encoding=pikepdf.Name("/Identity-H"),
                DescendantFonts=[descendant],
                ToUnicode=self.unicode,
            )
        )

    def overlay(self, target, commands, width, height, rotation):
        # A real page object lets pikepdf handle crop boxes and all quarter turns.
        page = pikepdf.Page(
            self.pdf.make_indirect(
                pikepdf.Dictionary(
                    Type=pikepdf.Name.Page,
                    MediaBox=[0, 0, width, height],
                    Rotate=(-rotation) % 360,
                    Resources=pikepdf.Dictionary(
                        Font=pikepdf.Dictionary(OCR=self.font)
                    ),
                    Contents=self.pdf.make_stream("\n".join(commands).encode("ascii")),
                )
            )
        )
        target.add_overlay(page, pikepdf.Rectangle(target.cropbox))

    def finish(self):
        if not self.codes:
            return
        # Glyph 5 is the font's blank, 1000-unit-wide glyph; text remains invisible.
        self.glyphs.write(b"\x00\x00" + b"\x00\x05" * len(self.codes))
        cmap = [
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap",
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
            "/CMapName /EditableScan-UCS def /CMapType 2 def",
            "1 begincodespacerange <0000> <FFFF> endcodespacerange",
        ]
        items = list(self.codes.items())
        for start in range(0, len(items), 100):
            group = items[start : start + 100]
            cmap.append(f"{len(group)} beginbfchar")
            cmap.extend(
                f"<{code:04X}> <{char.encode('utf-16-be').hex()}>"
                for char, code in group
            )
            cmap.append("endbfchar")
        cmap.append("endcmap CMapName currentdict /CMap defineresource pop end end")
        self.unicode.write("\n".join(cmap).encode("ascii"))

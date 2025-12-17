"""Color palette definitions for dithering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Palette:
    """A named color palette."""
    name: str
    colors: List[Tuple[int, int, int]]
    category: str = "Custom"
    description: str = ""


class PaletteLibrary:
    """Registry of color palettes organized by category."""

    def __init__(self) -> None:
        self._palettes: Dict[str, Palette] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in palettes."""
        # Monochrome
        self.register(Palette(
            "1-Bit Black & White",
            [(0, 0, 0), (255, 255, 255)],
            "Monochrome",
            "Pure black and white"
        ))
        self.register(Palette(
            "2-Bit Grayscale",
            [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)],
            "Monochrome",
            "4-level grayscale"
        ))
        self.register(Palette(
            "4-Bit Grayscale",
            [(i * 17, i * 17, i * 17) for i in range(16)],
            "Monochrome",
            "16-level grayscale"
        ))

        # Retro Computing
        self.register(Palette(
            "Game Boy",
            [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
            "Retro",
            "Original Game Boy green palette"
        ))
        self.register(Palette(
            "Game Boy Pocket",
            [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)],
            "Retro",
            "Game Boy Pocket grayscale"
        ))
        self.register(Palette(
            "CGA Mode 4 High",
            [(0, 0, 0), (85, 255, 255), (255, 85, 255), (255, 255, 255)],
            "Retro",
            "CGA cyan-magenta-white palette"
        ))
        self.register(Palette(
            "CGA Mode 4 Low",
            [(0, 0, 0), (0, 170, 0), (170, 0, 0), (170, 85, 0)],
            "Retro",
            "CGA green-red-brown palette"
        ))
        self.register(Palette(
            "Commodore 64",
            [
                (0, 0, 0), (255, 255, 255), (136, 57, 50), (103, 182, 189),
                (139, 63, 150), (85, 160, 73), (64, 49, 141), (191, 206, 114),
                (139, 84, 41), (87, 66, 0), (184, 105, 98), (80, 80, 80),
                (120, 120, 120), (148, 224, 137), (120, 105, 196), (159, 159, 159)
            ],
            "Retro",
            "Full Commodore 64 palette"
        ))
        self.register(Palette(
            "NES",
            [
                (0, 0, 0), (252, 252, 252), (188, 188, 188), (124, 124, 124),
                (164, 228, 252), (60, 188, 252), (0, 120, 248), (0, 0, 252),
                (216, 248, 120), (88, 216, 84), (0, 168, 0), (0, 168, 68),
                (248, 216, 120), (248, 184, 0), (248, 120, 88), (248, 56, 0)
            ],
            "Retro",
            "Nintendo NES color subset"
        ))

        # Artistic
        self.register(Palette(
            "Sepia",
            [(44, 33, 24), (90, 68, 49), (146, 111, 80), (201, 155, 113), (239, 207, 175)],
            "Artistic",
            "Warm sepia tones"
        ))
        self.register(Palette(
            "Blueprint",
            [(0, 18, 51), (0, 45, 128), (0, 82, 204), (51, 133, 255), (153, 194, 255)],
            "Artistic",
            "Technical blueprint blue"
        ))
        self.register(Palette(
            "Sunset",
            [(25, 10, 40), (80, 20, 60), (180, 60, 50), (255, 140, 60), (255, 220, 120)],
            "Artistic",
            "Warm sunset gradient"
        ))
        self.register(Palette(
            "Cyberpunk",
            [(10, 0, 20), (60, 0, 100), (255, 0, 128), (0, 255, 255), (255, 255, 0)],
            "Artistic",
            "Neon cyberpunk colors"
        ))
        self.register(Palette(
            "Vapor Wave",
            [(255, 113, 206), (185, 103, 255), (1, 205, 254), (5, 255, 161), (255, 251, 150)],
            "Artistic",
            "Retro vapor wave aesthetic"
        ))
        self.register(Palette(
            "Forest",
            [(27, 38, 29), (58, 90, 64), (88, 129, 87), (163, 177, 138), (218, 215, 205)],
            "Artistic",
            "Natural forest greens"
        ))
        self.register(Palette(
            "Ocean",
            [(3, 4, 94), (0, 119, 182), (0, 180, 216), (144, 224, 239), (202, 240, 248)],
            "Artistic",
            "Deep ocean blues"
        ))
        
        # Modern Design
        self.register(Palette(
            "Minimal Dark",
            [(9, 9, 11), (39, 39, 42), (82, 82, 91), (161, 161, 170), (250, 250, 250)],
            "Modern",
            "Clean minimal dark palette"
        ))
        self.register(Palette(
            "Minimal Light",
            [(250, 250, 250), (228, 228, 231), (161, 161, 170), (82, 82, 91), (24, 24, 27)],
            "Modern",
            "Clean minimal light palette"
        ))
        self.register(Palette(
            "Indigo",
            [(30, 27, 75), (67, 56, 202), (99, 102, 241), (165, 180, 252), (224, 231, 255)],
            "Modern",
            "Modern indigo gradient"
        ))
        self.register(Palette(
            "Emerald",
            [(6, 78, 59), (16, 185, 129), (52, 211, 153), (167, 243, 208), (236, 253, 245)],
            "Modern",
            "Fresh emerald tones"
        ))
        self.register(Palette(
            "Rose",
            [(136, 19, 55), (225, 29, 72), (251, 113, 133), (253, 164, 175), (255, 228, 230)],
            "Modern",
            "Soft rose gradient"
        ))
        
        # Print & Halftone
        self.register(Palette(
            "CMYK",
            [(0, 0, 0), (0, 174, 239), (236, 0, 140), (255, 242, 0), (255, 255, 255)],
            "Print",
            "Classic CMYK printing"
        ))
        self.register(Palette(
            "Newspaper",
            [(0, 0, 0), (255, 255, 255)],
            "Print",
            "Black and white newsprint"
        ))
        self.register(Palette(
            "Risograph Blue",
            [(0, 120, 191), (255, 255, 255)],
            "Print",
            "Risograph blue ink"
        ))
        self.register(Palette(
            "Risograph Red",
            [(255, 72, 72), (255, 255, 255)],
            "Print",
            "Risograph red ink"
        ))
        
        # Gradient palettes
        self.register(Palette(
            "Sunset Gradient",
            [(255, 94, 77), (255, 154, 62), (255, 206, 84), (255, 238, 173)],
            "Gradient",
            "Warm sunset colors"
        ))
        self.register(Palette(
            "Ocean Gradient",
            [(0, 63, 92), (44, 107, 142), (87, 157, 179), (155, 203, 200)],
            "Gradient",
            "Cool ocean depths"
        ))
        self.register(Palette(
            "Neon Gradient",
            [(255, 0, 128), (255, 0, 255), (128, 0, 255), (0, 128, 255)],
            "Gradient",
            "Vibrant neon colors"
        ))
        self.register(Palette(
            "Earth Gradient",
            [(69, 49, 34), (122, 90, 61), (179, 147, 105), (214, 196, 166)],
            "Gradient",
            "Natural earth tones"
        ))
        self.register(Palette(
            "Aurora Gradient",
            [(0, 28, 48), (0, 97, 97), (0, 194, 142), (128, 255, 212)],
            "Gradient",
            "Northern lights colors"
        ))
        
        # Film/Photo palettes
        self.register(Palette(
            "Kodak Portra",
            [(255, 248, 240), (255, 223, 196), (238, 180, 143), (189, 126, 103), (82, 67, 65)],
            "Film",
            "Kodak Portra 400 film emulation"
        ))
        self.register(Palette(
            "Fuji Velvia",
            [(0, 45, 90), (0, 100, 130), (45, 170, 130), (215, 210, 130), (255, 160, 90)],
            "Film",
            "Fuji Velvia vivid film"
        ))
        self.register(Palette(
            "Kodak Gold",
            [(255, 235, 180), (255, 200, 120), (235, 160, 80), (180, 100, 60), (60, 40, 30)],
            "Film",
            "Kodak Gold warm tones"
        ))
        self.register(Palette(
            "Ilford HP5",
            [(25, 25, 25), (70, 70, 70), (128, 128, 128), (190, 190, 190), (245, 245, 245)],
            "Film",
            "Ilford HP5 B&W film"
        ))
        self.register(Palette(
            "Cinestill 800T",
            [(20, 30, 50), (60, 70, 110), (150, 130, 140), (220, 200, 190), (255, 240, 230)],
            "Film",
            "Cinestill 800T tungsten film"
        ))
        
        # Duotone palettes
        self.register(Palette(
            "Duotone Blue",
            [(15, 23, 42), (59, 130, 246)],
            "Duotone",
            "Classic blue duotone"
        ))
        self.register(Palette(
            "Duotone Purple",
            [(30, 15, 45), (168, 85, 247)],
            "Duotone",
            "Rich purple duotone"
        ))
        self.register(Palette(
            "Duotone Orange",
            [(45, 20, 10), (249, 115, 22)],
            "Duotone",
            "Warm orange duotone"
        ))
        self.register(Palette(
            "Duotone Pink",
            [(45, 15, 30), (236, 72, 153)],
            "Duotone",
            "Vibrant pink duotone"
        ))
        self.register(Palette(
            "Duotone Teal",
            [(10, 35, 35), (20, 184, 166)],
            "Duotone",
            "Cool teal duotone"
        ))
        
        # Professional/Design palettes
        self.register(Palette(
            "Material Blue Grey",
            [(38, 50, 56), (69, 90, 100), (96, 125, 139), (144, 164, 174), (207, 216, 220)],
            "Design",
            "Material Design Blue Grey"
        ))
        self.register(Palette(
            "Material Deep Purple",
            [(49, 27, 146), (103, 58, 183), (149, 117, 205), (179, 157, 219), (209, 196, 233)],
            "Design",
            "Material Design Deep Purple"
        ))
        self.register(Palette(
            "iOS Light",
            [(255, 255, 255), (242, 242, 247), (199, 199, 204), (142, 142, 147), (44, 44, 46)],
            "Design",
            "Apple iOS light theme"
        ))
        self.register(Palette(
            "iOS Dark",
            [(0, 0, 0), (28, 28, 30), (44, 44, 46), (99, 99, 102), (174, 174, 178)],
            "Design",
            "Apple iOS dark theme"
        ))
        self.register(Palette(
            "Tailwind Slate",
            [(15, 23, 42), (51, 65, 85), (100, 116, 139), (148, 163, 184), (226, 232, 240)],
            "Design",
            "Tailwind CSS Slate"
        ))
        
        # Vintage/Retro palettes
        self.register(Palette(
            "Polaroid",
            [(255, 255, 250), (250, 240, 220), (220, 190, 160), (140, 110, 90), (50, 40, 35)],
            "Vintage",
            "Polaroid instant film look"
        ))
        self.register(Palette(
            "Faded Print",
            [(240, 235, 225), (200, 180, 160), (150, 130, 115), (100, 85, 75), (55, 50, 45)],
            "Vintage",
            "Aged newspaper print"
        ))
        self.register(Palette(
            "70s Retro",
            [(242, 229, 194), (216, 164, 86), (189, 99, 64), (140, 69, 57), (73, 45, 42)],
            "Vintage",
            "1970s color palette"
        ))
        self.register(Palette(
            "80s Neon",
            [(15, 15, 35), (255, 0, 128), (0, 255, 255), (255, 255, 0), (255, 128, 0)],
            "Vintage",
            "1980s neon aesthetic"
        ))

    def register(self, palette: Palette) -> None:
        """Register a palette."""
        self._palettes[palette.name] = palette

    def get(self, name: str) -> Optional[Palette]:
        """Get a palette by name."""
        return self._palettes.get(name)

    def all(self) -> List[Palette]:
        """Get all palettes."""
        return list(self._palettes.values())

    def categories(self) -> Dict[str, List[Palette]]:
        """Get palettes grouped by category."""
        result: Dict[str, List[Palette]] = {}
        for palette in self._palettes.values():
            if palette.category not in result:
                result[palette.category] = []
            result[palette.category].append(palette)
        return result

    def names(self) -> List[str]:
        """Get all palette names."""
        return list(self._palettes.keys())

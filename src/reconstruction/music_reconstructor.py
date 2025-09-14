"""
Music Reconstruction Module
==========================

Groups detected symbols into meaningful musical constructs and reconstructs
the musical meaning including pitch, rhythm, voices, and measures.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Import detection types
from ..detection.symbol_detector import DetectedSymbol
from ..detection.staff_detector import Staff

logger = logging.getLogger(__name__)


@dataclass
class MusicalElement:
    """Base class for all musical elements."""
    element_type: str = ""
    confidence: float = 0.0
    x_position: float = 0.0
    y_position: float = 0.0
    staff_index: int = 0
    measure_index: Optional[int] = None


@dataclass
class Note(MusicalElement):
    """Represents a musical note."""
    pitch: str = ""
    duration: float = 1.0  # In quarter note units
    voice: int = 1
    is_rest: bool = False
    dotted: bool = False
    accidental: Optional[str] = None
    tied_to_next: bool = False
    tied_from_prev: bool = False
    stem_direction: Optional[str] = None  # 'up' or 'down'
    beam_group: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        self.element_type = 'note'


@dataclass
class Chord(MusicalElement):
    """Represents a chord (multiple notes played simultaneously)."""
    notes: List[Note] = field(default_factory=list)
    duration: float = 1.0
    voice: int = 1
    
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        self.element_type = 'chord'


@dataclass
class TimeSignature(MusicalElement):
    """Represents a time signature."""
    numerator: int = 4
    denominator: int = 4
    
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        self.element_type = 'time_signature'


@dataclass
class KeySignature(MusicalElement):
    """Represents a key signature."""
    sharps: int = 0  # Positive for sharps, negative for flats
    key: str = "C major"     # e.g., "C major", "A minor"
    
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        self.element_type = 'key_signature'


@dataclass
class Clef(MusicalElement):
    """Represents a clef."""
    clef_type: str = "treble"  # 'treble', 'bass', 'alto', 'tenor'
    
    def __post_init__(self):
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        self.element_type = 'clef'


@dataclass
class Measure:
    """Represents a musical measure."""
    index: int = 0
    elements: List[MusicalElement] = field(default_factory=list)
    time_signature: Optional[TimeSignature] = None
    key_signature: Optional[KeySignature] = None
    x_start: float = 0
    x_end: float = 0
    staff_index: int = 0


@dataclass
class Voice:
    """Represents a voice within a staff."""
    voice_number: int = 1
    elements: List[MusicalElement] = field(default_factory=list)
    staff_index: int = 0


class MusicReconstructor:
    """
    Reconstructs musical meaning from detected symbols.
    """
    
    # Note duration mappings (in quarter note units)
    DURATION_MAP = {
        'whole_note': 4.0,
        'half_note': 2.0,
        'quarter_note': 1.0,
        'eighth_note': 0.5,
        'sixteenth_note': 0.25,
        'thirty_second_note': 0.125,
        'whole_rest': 4.0,
        'half_rest': 2.0,
        'quarter_rest': 1.0,
        'eighth_rest': 0.5,
        'sixteenth_rest': 0.25
    }
    
    # Pitch mappings for different clefs
    CLEF_PITCH_MAPS = {
        'treble': {
            # Line positions (staff line = 0, 1, 2, 3, 4 from bottom to top)
            # Space positions between lines
            -4: 'C4', -3: 'D4', -2: 'E4', -1: 'F4', 0: 'G4',   # Ledger lines and spaces below
            1: 'A4', 2: 'B4', 3: 'C5', 4: 'D5', 5: 'E5',      # Staff lines and spaces
            6: 'F5', 7: 'G5', 8: 'A5', 9: 'B5', 10: 'C6',     # Staff lines and spaces
            11: 'D6', 12: 'E6', 13: 'F6', 14: 'G6', 15: 'A6'  # Ledger lines above
        },
        'bass': {
            -4: 'E2', -3: 'F2', -2: 'G2', -1: 'A2', 0: 'B2',
            1: 'C3', 2: 'D3', 3: 'E3', 4: 'F3', 5: 'G3',
            6: 'A3', 7: 'B3', 8: 'C4', 9: 'D4', 10: 'E4',
            11: 'F4', 12: 'G4', 13: 'A4', 14: 'B4', 15: 'C5'
        }
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize the music reconstructor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.pitch_tolerance = self.config.get('pitch_tolerance', 5)
        self.rhythm_quantization = self.config.get('rhythm_quantization', True)
        self.beam_detection = self.config.get('beam_detection', True)
        self.voice_separation = self.config.get('voice_separation', True)
    
    def reconstruct_music(self, symbols: List[DetectedSymbol], 
                         staff_info: List[Staff]) -> Dict:
        """
        Reconstruct musical meaning from detected symbols.
        
        Args:
            symbols: List of detected symbols
            staff_info: Information about detected staves
            
        Returns:
            Dictionary containing reconstructed musical elements
        """
        logger.info("Starting music reconstruction")
        
        # Step 1: Group symbols by staff
        staff_symbols = self._group_symbols_by_staff(symbols, staff_info)
        
        # Step 2: Detect clefs, time signatures, and key signatures
        musical_context = self._extract_musical_context(staff_symbols, staff_info)
        
        # Step 3: Process each staff
        staves_data = []
        for staff_idx, (staff, staff_symbol_list) in enumerate(zip(staff_info, staff_symbols)):
            logger.info(f"Processing staff {staff_idx + 1}")
            
            # Convert symbols to musical elements
            elements = self._symbols_to_musical_elements(
                staff_symbol_list, staff, staff_idx, musical_context
            )
            
            # Group elements into measures
            measures = self._group_into_measures(elements, staff_symbol_list)
            
            # Separate voices if multiple voices detected
            if self.voice_separation:
                voices = self._separate_voices(elements, measures)
            else:
                voices = [Voice(voice_number=1, elements=elements, staff_index=staff_idx)]
            
            staff_data = {
                'staff_index': staff_idx,
                'staff_info': staff,
                'measures': measures,
                'voices': voices,
                'elements': elements,
                'clef': musical_context.get(f'staff_{staff_idx}_clef'),
                'key_signature': musical_context.get(f'staff_{staff_idx}_key_signature'),
                'time_signature': musical_context.get(f'staff_{staff_idx}_time_signature')
            }
            
            staves_data.append(staff_data)
        
        # Step 4: Align measures across staves
        aligned_measures = self._align_measures_across_staves(staves_data)
        
        # Step 5: Apply musical intelligence and cleanup
        final_result = self._apply_musical_intelligence(staves_data, aligned_measures)
        
        logger.info("Music reconstruction completed")
        return final_result
    
    def _group_symbols_by_staff(self, symbols: List[DetectedSymbol], 
                               staff_info: List[Staff]) -> List[List[DetectedSymbol]]:
        """
        Group symbols by which staff they belong to.
        
        Args:
            symbols: List of all detected symbols
            staff_info: Information about staves
            
        Returns:
            List of symbol lists, one for each staff
        """
        staff_symbols = [[] for _ in staff_info]
        
        for symbol in symbols:
            # Determine which staff this symbol belongs to
            symbol_y = symbol.center[1]
            best_staff_idx = None
            min_distance = float('inf')
            
            for staff_idx, staff in enumerate(staff_info):
                # Calculate distance to staff center
                staff_center = (staff.y_top + staff.y_bottom) / 2
                distance = abs(symbol_y - staff_center)
                
                # Also check if symbol is within staff boundaries (with some padding)
                staff_height = staff.y_bottom - staff.y_top
                padding = staff_height * 0.5  # 50% padding above and below
                
                if (staff.y_top - padding <= symbol_y <= staff.y_bottom + padding and 
                    distance < min_distance):
                    min_distance = distance
                    best_staff_idx = staff_idx
            
            if best_staff_idx is not None:
                staff_symbols[best_staff_idx].append(symbol)
        
        # Sort symbols within each staff by x position
        for staff_symbol_list in staff_symbols:
            staff_symbol_list.sort(key=lambda s: s.center[0])
        
        return staff_symbols
    
    def _extract_musical_context(self, staff_symbols: List[List[DetectedSymbol]], 
                                staff_info: List[Staff]) -> Dict:
        """
        Extract clefs, time signatures, and key signatures from symbols.
        
        Args:
            staff_symbols: Symbols grouped by staff
            staff_info: Staff information
            
        Returns:
            Dictionary with musical context for each staff
        """
        context = {}
        
        for staff_idx, symbol_list in enumerate(staff_symbols):
            # Find clef (usually at the beginning)
            clef = None
            for symbol in symbol_list:
                if 'clef' in symbol.class_name:
                    clef_type = symbol.class_name.replace('_clef', '')
                    clef = Clef(
                        clef_type=clef_type,
                        confidence=symbol.confidence,
                        x_position=symbol.center[0],
                        y_position=symbol.center[1],
                        staff_index=staff_idx
                    )
                    break
            
            # Default to treble clef if none found
            if not clef:
                clef = Clef(
                    clef_type='treble',
                    confidence=0.5,  # Default confidence
                    x_position=0,
                    y_position=staff_info[staff_idx].lines[2].y_position,
                    staff_index=staff_idx
                )
            
            context[f'staff_{staff_idx}_clef'] = clef
            
            # Find time signature
            time_sig = None
            for symbol in symbol_list:
                if 'time_signature' in symbol.class_name:
                    # Parse time signature from class name
                    if 'common' in symbol.class_name:
                        numerator, denominator = 4, 4
                    elif 'cut' in symbol.class_name:
                        numerator, denominator = 2, 2
                    else:
                        # Extract numbers from class name (e.g., "time_signature_3_4")
                        parts = symbol.class_name.split('_')
                        if len(parts) >= 4:
                            try:
                                numerator = int(parts[2])
                                denominator = int(parts[3])
                            except ValueError:
                                numerator, denominator = 4, 4
                        else:
                            numerator, denominator = 4, 4
                    
                    time_sig = TimeSignature(
                        numerator=numerator,
                        denominator=denominator,
                        confidence=symbol.confidence,
                        x_position=symbol.center[0],
                        y_position=symbol.center[1],
                        staff_index=staff_idx
                    )
                    break
            
            # Default to 4/4 if none found
            if not time_sig:
                time_sig = TimeSignature(
                    numerator=4,
                    denominator=4,
                    confidence=0.5,
                    x_position=50,  # Default position
                    y_position=staff_info[staff_idx].lines[2].y_position,
                    staff_index=staff_idx
                )
            
            context[f'staff_{staff_idx}_time_signature'] = time_sig
            
            # Key signature detection (simplified - just count sharps/flats)
            key_sig = KeySignature(
                sharps=0,  # TODO: Implement key signature detection
                key="C major",
                confidence=0.5,
                x_position=30,
                y_position=staff_info[staff_idx].lines[2].y_position,
                staff_index=staff_idx
            )
            
            context[f'staff_{staff_idx}_key_signature'] = key_sig
        
        return context
    
    def _symbols_to_musical_elements(self, symbols: List[DetectedSymbol], 
                                   staff: Staff, staff_idx: int, 
                                   context: Dict) -> List[MusicalElement]:
        """
        Convert detected symbols to musical elements.
        
        Args:
            symbols: Symbols for this staff
            staff: Staff information
            staff_idx: Staff index
            context: Musical context
            
        Returns:
            List of musical elements
        """
        elements = []
        clef = context.get(f'staff_{staff_idx}_clef')
        
        for symbol in symbols:
            if 'note' in symbol.class_name or 'rest' in symbol.class_name:
                # Create note or rest
                duration = self.DURATION_MAP.get(symbol.class_name, 1.0)
                
                # Apply dotted note logic
                dotted = False
                if symbol.additional_data and symbol.additional_data.get('augmentation_dots', 0) > 0:
                    dotted = True
                    duration *= 1.5  # Dotted note is 1.5x the original duration
                
                # Determine pitch
                pitch = self._calculate_pitch(symbol, staff, clef)
                
                # Check for accidentals
                accidental = None
                if symbol.additional_data and 'accidental' in symbol.additional_data:
                    accidental = symbol.additional_data['accidental']
                
                note = Note(
                    pitch=pitch or 'C4',  # Default pitch if calculation fails
                    duration=duration,
                    is_rest='rest' in symbol.class_name,
                    dotted=dotted,
                    accidental=accidental,
                    confidence=symbol.confidence,
                    x_position=symbol.center[0],
                    y_position=symbol.center[1],
                    staff_index=staff_idx
                )
                
                elements.append(note)
            
            elif symbol.class_name in ['treble_clef', 'bass_clef', 'alto_clef', 'tenor_clef']:
                # Clef already handled in context extraction
                pass
            
            elif 'time_signature' in symbol.class_name:
                # Time signature already handled in context extraction
                pass
        
        return elements
    
    def _calculate_pitch(self, symbol: DetectedSymbol, staff: Staff, clef: Clef) -> Optional[str]:
        """
        Calculate the pitch of a note based on its position on the staff.
        
        Args:
            symbol: Detected symbol
            staff: Staff information
            clef: Clef for this staff
            
        Returns:
            Pitch string (e.g., "C4") or None if calculation fails
        """
        if not clef or clef.clef_type not in self.CLEF_PITCH_MAPS:
            return None
        
        # Calculate position relative to staff lines
        symbol_y = symbol.center[1]
        line_spacing = staff.line_spacing
        
        # Find the closest staff line
        line_distances = [abs(symbol_y - line.y_position) for line in staff.lines]
        closest_line_idx = np.argmin(line_distances)
        closest_line_y = staff.lines[closest_line_idx].y_position
        
        # Calculate steps from the middle line (index 2)
        steps_from_middle_line = closest_line_idx - 2
        
        # Adjust for position relative to the closest line
        vertical_offset = symbol_y - closest_line_y
        half_steps = round(vertical_offset / (line_spacing / 2))
        
        # Total position in half-steps from middle line
        total_steps = steps_from_middle_line * 2 + half_steps
        
        # Look up pitch in clef map
        pitch_map = self.CLEF_PITCH_MAPS[clef.clef_type]
        
        # Adjust for the fact that our map uses line spacing as the unit
        map_position = -total_steps + 5  # Adjust to match map indexing
        
        return pitch_map.get(map_position, 'C4')  # Default to C4 if not found
    
    def _group_into_measures(self, elements: List[MusicalElement], 
                           symbols: List[DetectedSymbol]) -> List[Measure]:
        """
        Group musical elements into measures based on barlines.
        
        Args:
            elements: Musical elements for this staff
            symbols: Original detected symbols
            
        Returns:
            List of measures
        """
        # Find barlines
        barlines = [s for s in symbols if 'barline' in s.class_name]
        barlines.sort(key=lambda b: b.center[0])
        
        # If no barlines found, create one measure with all elements
        if not barlines:
            measure = Measure(
                index=0,
                elements=elements,
                x_start=min(e.x_position for e in elements) if elements else 0,
                x_end=max(e.x_position for e in elements) if elements else 100,
                staff_index=elements[0].staff_index if elements else 0
            )
            return [measure]
        
        measures = []
        current_measure_elements = []
        current_x_start = 0
        measure_index = 0
        
        # Sort elements by x position
        sorted_elements = sorted(elements, key=lambda e: e.x_position)
        element_index = 0
        
        for barline in barlines:
            barline_x = barline.center[0]
            
            # Collect elements before this barline
            while element_index < len(sorted_elements) and sorted_elements[element_index].x_position < barline_x:
                current_measure_elements.append(sorted_elements[element_index])
                element_index += 1
            
            # Create measure if we have elements
            if current_measure_elements:
                measure = Measure(
                    index=measure_index,
                    elements=current_measure_elements,
                    x_start=current_x_start,
                    x_end=barline_x,
                    staff_index=current_measure_elements[0].staff_index if current_measure_elements else 0
                )
                measures.append(measure)
                measure_index += 1
                current_measure_elements = []
            
            current_x_start = barline_x
        
        # Handle remaining elements after the last barline
        remaining_elements = sorted_elements[element_index:]
        if remaining_elements:
            measure = Measure(
                index=measure_index,
                elements=remaining_elements,
                x_start=current_x_start,
                x_end=max(e.x_position for e in remaining_elements),
                staff_index=remaining_elements[0].staff_index
            )
            measures.append(measure)
        
        return measures
    
    def _separate_voices(self, elements: List[MusicalElement], 
                        measures: List[Measure]) -> List[Voice]:
        """
        Separate elements into different voices based on vertical position and timing.
        
        Args:
            elements: Musical elements to separate
            measures: Measures for context
            
        Returns:
            List of voices
        """
        # Simple voice separation based on Y position
        # More sophisticated algorithms would consider stem direction, timing, etc.
        
        note_elements = [e for e in elements if isinstance(e, Note) and not e.is_rest]
        
        if len(note_elements) < 2:
            # Not enough notes for voice separation
            voice = Voice(
                voice_number=1,
                elements=elements,
                staff_index=elements[0].staff_index if elements else 0
            )
            return [voice]
        
        # Group notes by similar Y positions (voices tend to stay in similar ranges)
        y_positions = [note.y_position for note in note_elements]
        
        # Use K-means-like clustering to separate voices
        # For simplicity, assume maximum 2 voices per staff
        y_mean = np.mean(y_positions)
        
        voice1_elements = []
        voice2_elements = []
        
        for element in elements:
            if isinstance(element, Note):
                if element.y_position <= y_mean:
                    element.voice = 1
                    voice1_elements.append(element)
                else:
                    element.voice = 2
                    voice2_elements.append(element)
            else:
                # Non-note elements go to voice 1 by default
                voice1_elements.append(element)
        
        voices = []
        if voice1_elements:
            voices.append(Voice(
                voice_number=1,
                elements=voice1_elements,
                staff_index=voice1_elements[0].staff_index
            ))
        
        if voice2_elements:
            voices.append(Voice(
                voice_number=2,
                elements=voice2_elements,
                staff_index=voice2_elements[0].staff_index
            ))
        
        return voices
    
    def _align_measures_across_staves(self, staves_data: List[Dict]) -> List[List[Measure]]:
        """
        Align measures across different staves.
        
        Args:
            staves_data: Data for each staff
            
        Returns:
            List of measure groups, each containing one measure from each staff
        """
        if not staves_data:
            return []
        
        # Find the maximum number of measures across all staves
        max_measures = max(len(staff_data['measures']) for staff_data in staves_data)
        
        aligned_measures = []
        
        for measure_idx in range(max_measures):
            measure_group = []
            
            for staff_data in staves_data:
                if measure_idx < len(staff_data['measures']):
                    measure_group.append(staff_data['measures'][measure_idx])
                else:
                    # Create empty measure if this staff doesn't have this measure
                    empty_measure = Measure(
                        index=measure_idx,
                        elements=[],
                        staff_index=staff_data['staff_index']
                    )
                    measure_group.append(empty_measure)
            
            aligned_measures.append(measure_group)
        
        return aligned_measures
    
    def _apply_musical_intelligence(self, staves_data: List[Dict], 
                                  aligned_measures: List[List[Measure]]) -> Dict:
        """
        Apply musical intelligence to clean up and enhance the reconstruction.
        
        Args:
            staves_data: Data for each staff
            aligned_measures: Aligned measures across staves
            
        Returns:
            Final reconstructed musical data
        """
        # Quantize rhythms if enabled
        if self.rhythm_quantization:
            for staff_data in staves_data:
                self._quantize_rhythms(staff_data['voices'])
        
        # Detect and mark tied notes
        for staff_data in staves_data:
            self._detect_tied_notes(staff_data['voices'])
        
        # Compile final result
        result = {
            'staves': staves_data,
            'aligned_measures': aligned_measures,
            'metadata': {
                'num_staves': len(staves_data),
                'num_measures': len(aligned_measures),
                'reconstruction_config': self.config
            }
        }
        
        return result
    
    def _quantize_rhythms(self, voices: List[Voice]):
        """
        Quantize note durations to standard musical values.
        
        Args:
            voices: List of voices to quantize
        """
        standard_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
        
        for voice in voices:
            for element in voice.elements:
                if isinstance(element, Note):
                    # Find closest standard duration
                    closest_duration = min(standard_durations, 
                                         key=lambda d: abs(d - element.duration))
                    element.duration = closest_duration
    
    def _detect_tied_notes(self, voices: List[Voice]):
        """
        Detect and mark tied notes.
        
        Args:
            voices: List of voices to analyze
        """
        for voice in voices:
            notes = [e for e in voice.elements if isinstance(e, Note) and not e.is_rest]
            
            for i in range(len(notes) - 1):
                current_note = notes[i]
                next_note = notes[i + 1]
                
                # Check if notes have same pitch and are close horizontally
                if (current_note.pitch == next_note.pitch and
                    abs(next_note.x_position - current_note.x_position) < 100):  # Threshold in pixels
                    current_note.tied_to_next = True
                    next_note.tied_from_prev = True


def create_reconstruction_package_init():
    """Create __init__.py for reconstruction package."""
    return '''"""
Reconstruction Package
=====================

Musical meaning reconstruction from detected symbols.
"""

from .music_reconstructor import (
    MusicReconstructor, MusicalElement, Note, Chord, 
    TimeSignature, KeySignature, Clef, Measure, Voice
)

__all__ = [
    'MusicReconstructor', 'MusicalElement', 'Note', 'Chord',
    'TimeSignature', 'KeySignature', 'Clef', 'Measure', 'Voice'
]
'''


if __name__ == "__main__":
    # Test function
    reconstructor = MusicReconstructor()
    print("Music reconstructor initialized")
    print(f"Duration map has {len(reconstructor.DURATION_MAP)} note types")
    print(f"Clef pitch maps available for: {list(reconstructor.CLEF_PITCH_MAPS.keys())}")
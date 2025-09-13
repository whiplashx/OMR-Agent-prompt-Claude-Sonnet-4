"""
MusicXML Generator Module
========================

Generates MusicXML files from reconstructed musical elements.
Compatible with MuseScore and other music notation software.
Uses divisions=480 for precise timing representation.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional
import logging
from datetime import datetime

from ..reconstruction.music_reconstructor import (
    MusicalElement, Note, Chord, TimeSignature, KeySignature, 
    Clef, Measure, Voice
)

logger = logging.getLogger(__name__)


class MusicXMLGenerator:
    """
    Generates MusicXML format files from reconstructed musical data.
    """
    
    # MusicXML uses divisions to represent timing
    # divisions=480 allows for precise representation of various note values
    DIVISIONS = 480
    
    # Note duration mappings to divisions
    DURATION_TO_DIVISIONS = {
        4.0: DIVISIONS * 4,      # Whole note
        2.0: DIVISIONS * 2,      # Half note
        1.0: DIVISIONS,          # Quarter note
        0.5: DIVISIONS // 2,     # Eighth note
        0.25: DIVISIONS // 4,    # Sixteenth note
        0.125: DIVISIONS // 8,   # Thirty-second note
    }
    
    # Note type mappings
    DURATION_TO_TYPE = {
        4.0: 'whole',
        2.0: 'half',
        1.0: 'quarter',
        0.5: 'eighth',
        0.25: '16th',
        0.125: '32nd',
    }
    
    # Clef mappings
    CLEF_MAPPING = {
        'treble': ('G', 2),
        'bass': ('F', 4),
        'alto': ('C', 3),
        'tenor': ('C', 4)
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize the MusicXML generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.divisions = self.config.get('divisions', self.DIVISIONS)
        self.validate = self.config.get('validate', True)
        self.software_name = self.config.get('software_name', 'OMR Pipeline')
        self.software_version = self.config.get('software_version', '1.0.0')
        
        # Update duration mappings if custom divisions are used
        if self.divisions != self.DIVISIONS:
            self._update_duration_mappings()
    
    def _update_duration_mappings(self):
        """Update duration mappings for custom divisions."""
        scale_factor = self.divisions / self.DIVISIONS
        self.DURATION_TO_DIVISIONS = {
            duration: int(divisions * scale_factor)
            for duration, divisions in self.DURATION_TO_DIVISIONS.items()
        }
    
    def generate(self, musical_data: Dict, staff_info: List = None) -> str:
        """
        Generate MusicXML from musical data.
        
        Args:
            musical_data: Reconstructed musical data
            staff_info: Optional staff information
            
        Returns:
            MusicXML content as string
        """
        logger.info("Generating MusicXML")
        
        # Create root element
        root = ET.Element('score-partwise')
        root.set('version', '3.1')
        
        # Add header information
        self._add_header(root)
        
        # Add part list
        part_list = self._create_part_list(musical_data)
        root.append(part_list)
        
        # Add parts (staves)
        for staff_data in musical_data.get('staves', []):
            part = self._create_part(staff_data, musical_data.get('aligned_measures', []))
            root.append(part)
        
        # Convert to formatted XML string
        xml_string = self._format_xml(root)
        
        # Validate if requested
        if self.validate:
            self._validate_xml(xml_string)
        
        logger.info("MusicXML generation completed")
        return xml_string
    
    def _add_header(self, root: ET.Element):
        """
        Add header information to the MusicXML document.
        
        Args:
            root: Root XML element
        """
        # Work element
        work = ET.SubElement(root, 'work')
        work_title = ET.SubElement(work, 'work-title')
        work_title.text = 'OMR Transcription'
        
        # Identification
        identification = ET.SubElement(root, 'identification')
        
        # Creator
        creator = ET.SubElement(identification, 'creator')
        creator.set('type', 'software')
        creator.text = self.software_name
        
        # Encoding
        encoding = ET.SubElement(identification, 'encoding')
        
        software = ET.SubElement(encoding, 'software')
        software.text = f"{self.software_name} {self.software_version}"
        
        encoding_date = ET.SubElement(encoding, 'encoding-date')
        encoding_date.text = datetime.now().strftime('%Y-%m-%d')
        
        # Defaults
        defaults = ET.SubElement(root, 'defaults')
        scaling = ET.SubElement(defaults, 'scaling')
        
        millimeters = ET.SubElement(scaling, 'millimeters')
        millimeters.text = '7.05556'
        
        tenths = ET.SubElement(scaling, 'tenths')
        tenths.text = '40'
    
    def _create_part_list(self, musical_data: Dict) -> ET.Element:
        """
        Create the part-list element.
        
        Args:
            musical_data: Musical data
            
        Returns:
            Part list XML element
        """
        part_list = ET.Element('part-list')
        
        staves = musical_data.get('staves', [])
        
        for i, staff_data in enumerate(staves):
            score_part = ET.SubElement(part_list, 'score-part')
            score_part.set('id', f'P{i+1}')
            
            part_name = ET.SubElement(score_part, 'part-name')
            part_name.text = f'Staff {i+1}'
            
            # Add instrument information if available
            score_instrument = ET.SubElement(score_part, 'score-instrument')
            score_instrument.set('id', f'P{i+1}-I1')
            
            instrument_name = ET.SubElement(score_instrument, 'instrument-name')
            instrument_name.text = 'Piano'  # Default instrument
            
            midi_instrument = ET.SubElement(score_part, 'midi-instrument')
            midi_instrument.set('id', f'P{i+1}-I1')
            
            midi_channel = ET.SubElement(midi_instrument, 'midi-channel')
            midi_channel.text = str(i + 1)
            
            midi_program = ET.SubElement(midi_instrument, 'midi-program')
            midi_program.text = '1'  # Piano
        
        return part_list
    
    def _create_part(self, staff_data: Dict, aligned_measures: List[List[Measure]]) -> ET.Element:
        """
        Create a part element for a staff.
        
        Args:
            staff_data: Data for this staff
            aligned_measures: Aligned measures across all staves
            
        Returns:
            Part XML element
        """
        part = ET.Element('part')
        part.set('id', f'P{staff_data["staff_index"] + 1}')
        
        measures = staff_data.get('measures', [])
        
        for measure_idx, measure in enumerate(measures):
            measure_elem = self._create_measure(measure, measure_idx + 1, staff_data)
            part.append(measure_elem)
        
        return part
    
    def _create_measure(self, measure: Measure, measure_number: int, staff_data: Dict) -> ET.Element:
        """
        Create a measure element.
        
        Args:
            measure: Measure data
            measure_number: Measure number
            staff_data: Staff data for context
            
        Returns:
            Measure XML element
        """
        measure_elem = ET.Element('measure')
        measure_elem.set('number', str(measure_number))
        
        # Add attributes for first measure
        if measure_number == 1:
            attributes = self._create_attributes(staff_data)
            measure_elem.append(attributes)
        
        # Group elements by voice
        voices = self._group_elements_by_voice(measure.elements)
        
        # Process each voice
        for voice_num in sorted(voices.keys()):
            voice_elements = voices[voice_num]
            
            for element in voice_elements:
                if isinstance(element, Note):
                    note_elem = self._create_note(element, voice_num)
                    measure_elem.append(note_elem)
                elif isinstance(element, Chord):
                    chord_elems = self._create_chord(element, voice_num)
                    for chord_elem in chord_elems:
                        measure_elem.append(chord_elem)
        
        # Fill remaining duration with rests if needed
        self._fill_measure_with_rests(measure_elem, staff_data)
        
        return measure_elem
    
    def _create_attributes(self, staff_data: Dict) -> ET.Element:
        """
        Create attributes element for measure.
        
        Args:
            staff_data: Staff data
            
        Returns:
            Attributes XML element
        """
        attributes = ET.Element('attributes')
        
        # Divisions
        divisions = ET.SubElement(attributes, 'divisions')
        divisions.text = str(self.divisions)
        
        # Key signature
        key_sig = staff_data.get('key_signature')
        if key_sig:
            key = ET.SubElement(attributes, 'key')
            fifths = ET.SubElement(key, 'fifths')
            fifths.text = str(key_sig.sharps)
        
        # Time signature
        time_sig = staff_data.get('time_signature')
        if time_sig:
            time = ET.SubElement(attributes, 'time')
            beats = ET.SubElement(time, 'beats')
            beats.text = str(time_sig.numerator)
            beat_type = ET.SubElement(time, 'beat-type')
            beat_type.text = str(time_sig.denominator)
        
        # Clef
        clef_data = staff_data.get('clef')
        if clef_data and clef_data.clef_type in self.CLEF_MAPPING:
            clef = ET.SubElement(attributes, 'clef')
            sign, line = self.CLEF_MAPPING[clef_data.clef_type]
            
            clef_sign = ET.SubElement(clef, 'sign')
            clef_sign.text = sign
            
            clef_line = ET.SubElement(clef, 'line')
            clef_line.text = str(line)
        
        return attributes
    
    def _group_elements_by_voice(self, elements: List[MusicalElement]) -> Dict[int, List[MusicalElement]]:
        """
        Group musical elements by voice number.
        
        Args:
            elements: List of musical elements
            
        Returns:
            Dictionary mapping voice numbers to element lists
        """
        voices = {}
        
        for element in elements:
            voice_num = getattr(element, 'voice', 1)
            if voice_num not in voices:
                voices[voice_num] = []
            voices[voice_num].append(element)
        
        # Sort elements within each voice by x position
        for voice_elements in voices.values():
            voice_elements.sort(key=lambda e: e.x_position)
        
        return voices
    
    def _create_note(self, note: Note, voice_num: int) -> ET.Element:
        """
        Create a note element.
        
        Args:
            note: Note data
            voice_num: Voice number
            
        Returns:
            Note XML element
        """
        note_elem = ET.Element('note')
        
        # Rest or pitched note
        if note.is_rest:
            rest = ET.SubElement(note_elem, 'rest')
        else:
            pitch = self._create_pitch(note)
            note_elem.append(pitch)
        
        # Duration
        duration = ET.SubElement(note_elem, 'duration')
        duration.text = str(self._get_note_duration_divisions(note))
        
        # Voice
        voice = ET.SubElement(note_elem, 'voice')
        voice.text = str(voice_num)
        
        # Note type
        note_type = ET.SubElement(note_elem, 'type')
        note_type.text = self.DURATION_TO_TYPE.get(note.duration, 'quarter')
        
        # Dotted note
        if note.dotted:
            dot = ET.SubElement(note_elem, 'dot')
        
        # Accidental
        if note.accidental:
            accidental = ET.SubElement(note_elem, 'accidental')
            accidental.text = note.accidental
        
        # Tie
        if note.tied_to_next:
            tie = ET.SubElement(note_elem, 'tie')
            tie.set('type', 'start')
        
        if note.tied_from_prev:
            tie = ET.SubElement(note_elem, 'tie')
            tie.set('type', 'stop')
        
        return note_elem
    
    def _create_pitch(self, note: Note) -> ET.Element:
        """
        Create a pitch element from a note.
        
        Args:
            note: Note data
            
        Returns:
            Pitch XML element
        """
        pitch = ET.Element('pitch')
        
        # Parse pitch string (e.g., "C4", "F#5", "Bb3")
        pitch_str = note.pitch
        
        # Extract note name and octave
        if len(pitch_str) >= 2:
            step = pitch_str[0].upper()
            octave_start = 1
            
            # Check for accidental
            alter = 0
            if len(pitch_str) > 2 and pitch_str[1] in '#b':
                if pitch_str[1] == '#':
                    alter = 1
                elif pitch_str[1] == 'b':
                    alter = -1
                octave_start = 2
            
            # Extract octave
            try:
                octave = int(pitch_str[octave_start:])
            except (ValueError, IndexError):
                octave = 4  # Default octave
        else:
            step = 'C'
            alter = 0
            octave = 4
        
        # Create pitch elements
        step_elem = ET.SubElement(pitch, 'step')
        step_elem.text = step
        
        if alter != 0:
            alter_elem = ET.SubElement(pitch, 'alter')
            alter_elem.text = str(alter)
        
        octave_elem = ET.SubElement(pitch, 'octave')
        octave_elem.text = str(octave)
        
        return pitch
    
    def _create_chord(self, chord: Chord, voice_num: int) -> List[ET.Element]:
        """
        Create chord elements (multiple notes).
        
        Args:
            chord: Chord data
            voice_num: Voice number
            
        Returns:
            List of note XML elements
        """
        chord_elements = []
        
        for i, note in enumerate(chord.notes):
            note_elem = self._create_note(note, voice_num)
            
            # Add chord element to all but the first note
            if i > 0:
                chord_tag = ET.SubElement(note_elem, 'chord')
            
            chord_elements.append(note_elem)
        
        return chord_elements
    
    def _get_note_duration_divisions(self, note: Note) -> int:
        """
        Get the duration in divisions for a note.
        
        Args:
            note: Note data
            
        Returns:
            Duration in divisions
        """
        base_duration = self.DURATION_TO_DIVISIONS.get(note.duration, self.divisions)
        
        # Apply dotting
        if note.dotted:
            base_duration = int(base_duration * 1.5)
        
        return base_duration
    
    def _fill_measure_with_rests(self, measure_elem: ET.Element, staff_data: Dict):
        """
        Fill any remaining duration in a measure with rests.
        
        Args:
            measure_elem: Measure XML element
            staff_data: Staff data for time signature
        """
        # Calculate total duration that should be in the measure
        time_sig = staff_data.get('time_signature')
        if not time_sig:
            return
        
        # Expected duration in divisions
        expected_duration = (time_sig.numerator * self.divisions * 4) // time_sig.denominator
        
        # Calculate current duration from existing notes
        current_duration = 0
        for note_elem in measure_elem.findall('note'):
            duration_elem = note_elem.find('duration')
            if duration_elem is not None:
                current_duration += int(duration_elem.text)
        
        # Add rest if needed
        remaining_duration = expected_duration - current_duration
        if remaining_duration > 0:
            rest_elem = ET.Element('note')
            
            rest = ET.SubElement(rest_elem, 'rest')
            
            duration = ET.SubElement(rest_elem, 'duration')
            duration.text = str(remaining_duration)
            
            voice = ET.SubElement(rest_elem, 'voice')
            voice.text = '1'
            
            # Determine rest type
            rest_duration_quarters = remaining_duration / self.divisions
            rest_type = self.DURATION_TO_TYPE.get(rest_duration_quarters, 'quarter')
            
            note_type = ET.SubElement(rest_elem, 'type')
            note_type.text = rest_type
            
            measure_elem.append(rest_elem)
    
    def _format_xml(self, root: ET.Element) -> str:
        """
        Format XML with proper indentation.
        
        Args:
            root: Root XML element
            
        Returns:
            Formatted XML string
        """
        # Convert to string
        rough_string = ET.tostring(root, encoding='unicode')
        
        # Parse and pretty print
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent='  ')
        
        # Remove extra newlines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        
        return '\n'.join(lines)
    
    def _validate_xml(self, xml_string: str):
        """
        Validate the generated MusicXML.
        
        Args:
            xml_string: XML content to validate
        """
        try:
            # Basic XML validation
            ET.fromstring(xml_string)
            logger.info("MusicXML validation passed")
        except ET.ParseError as e:
            logger.error(f"MusicXML validation failed: {e}")
            raise ValueError(f"Invalid MusicXML generated: {e}")
    
    def save_to_file(self, xml_content: str, file_path: str):
        """
        Save MusicXML content to a file.
        
        Args:
            xml_content: MusicXML content
            file_path: Output file path
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        logger.info(f"MusicXML saved to {file_path}")


def test_musicxml_generator():
    """Test function for MusicXML generator."""
    from ..reconstruction.music_reconstructor import Note, TimeSignature, Clef, Measure
    
    # Create test data
    notes = [
        Note(
            pitch='C4',
            duration=1.0,
            confidence=0.9,
            x_position=100,
            y_position=200,
            staff_index=0
        ),
        Note(
            pitch='D4',
            duration=1.0,
            confidence=0.8,
            x_position=150,
            y_position=195,
            staff_index=0
        ),
        Note(
            pitch='E4',
            duration=2.0,
            confidence=0.85,
            x_position=200,
            y_position=190,
            staff_index=0
        ),
    ]
    
    measure = Measure(
        index=0,
        elements=notes,
        x_start=80,
        x_end=250,
        staff_index=0
    )
    
    time_sig = TimeSignature(
        numerator=4,
        denominator=4,
        confidence=0.9,
        x_position=50,
        y_position=200,
        staff_index=0
    )
    
    clef = Clef(
        clef_type='treble',
        confidence=0.95,
        x_position=20,
        y_position=200,
        staff_index=0
    )
    
    staff_data = {
        'staff_index': 0,
        'measures': [measure],
        'voices': [],
        'elements': notes,
        'clef': clef,
        'time_signature': time_sig,
        'key_signature': None
    }
    
    musical_data = {
        'staves': [staff_data],
        'aligned_measures': [[measure]]
    }
    
    # Generate MusicXML
    generator = MusicXMLGenerator()
    xml_content = generator.generate(musical_data)
    
    print("Generated MusicXML:")
    print(xml_content[:500] + "..." if len(xml_content) > 500 else xml_content)


if __name__ == "__main__":
    test_musicxml_generator()
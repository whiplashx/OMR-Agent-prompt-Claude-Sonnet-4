"""
Manual Correction UI
===================

Interactive Streamlit interface for reviewing and correcting OMR results.
Allows users to validate detections, correct errors, and refine output.
"""

import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd

# Import OMR modules
from ..omr_pipeline import OMRPipeline
from ..detection.symbol_detector import DetectedSymbol
from ..reconstruction.music_reconstructor import Note, MusicalElement


class OMRCorrectionUI:
    """
    Streamlit-based UI for manual correction of OMR results.
    """
    
    def __init__(self):
        """Initialize the correction UI."""
        self.pipeline = None
        self.current_results = None
        self.corrections = {}
        self.selected_symbol = None
        
        # Initialize session state
        if 'omr_results' not in st.session_state:
            st.session_state.omr_results = None
        if 'corrections' not in st.session_state:
            st.session_state.corrections = {}
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
    
    def run(self):
        """Run the Streamlit interface."""
        st.set_page_config(
            page_title="OMR Manual Correction",
            page_icon="🎵",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎵 OMR Manual Correction Interface")
        st.markdown("Review and correct optical music recognition results")
        
        # Sidebar for file upload and controls
        self._render_sidebar()
        
        # Main content area
        if st.session_state.omr_results is not None:
            self._render_main_interface()
        else:
            self._render_welcome_screen()
    
    def _render_sidebar(self):
        """Render the sidebar with controls and options."""
        st.sidebar.header("🎛️ Controls")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Sheet Music Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a sheet music image for OMR processing"
        )
        
        if uploaded_file is not None:
            # Process uploaded image
            if st.sidebar.button("🔍 Process Image", type="primary"):
                self._process_uploaded_image(uploaded_file)
        
        # Load existing results
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Load Existing Results")
        
        results_file = st.sidebar.file_uploader(
            "Load JSON Results",
            type=['json'],
            help="Load previously processed OMR results"
        )
        
        if results_file is not None:
            if st.sidebar.button("📥 Load Results"):
                self._load_existing_results(results_file)
        
        # Settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Settings")
        
        st.sidebar.checkbox("Show confidence scores", value=True, key="show_confidence")
        st.sidebar.checkbox("Show bounding boxes", value=True, key="show_bboxes")
        st.sidebar.checkbox("Highlight low confidence", value=True, key="highlight_low_conf")
        
        confidence_threshold = st.sidebar.slider(
            "Low confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="conf_threshold"
        )
        
        # Export options
        if st.session_state.omr_results is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("💾 Export")
            
            if st.sidebar.button("🎼 Export MusicXML"):
                self._export_musicxml()
            
            if st.sidebar.button("📊 Export Corrected JSON"):
                self._export_corrected_json()
    
    def _render_welcome_screen(self):
        """Render the welcome screen when no results are loaded."""
        st.markdown("""
        ## Welcome to the OMR Manual Correction Interface
        
        This tool allows you to review and correct the results of optical music recognition.
        
        ### How to use:
        1. **Upload an image**: Use the sidebar to upload a sheet music image
        2. **Process**: Click "Process Image" to run OMR analysis
        3. **Review**: Examine the detected symbols and their confidence scores
        4. **Correct**: Click on symbols to edit their properties
        5. **Export**: Save the corrected results as MusicXML or JSON
        
        ### Features:
        - 🔍 Interactive symbol visualization
        - ✏️ Click-to-edit symbol properties
        - 📊 Confidence analysis and filtering
        - 🎵 Real-time MusicXML preview
        - 💾 Export corrected results
        """)
        
        # Example images or demo data could go here
        st.info("👆 Start by uploading a sheet music image using the sidebar")
    
    def _render_main_interface(self):
        """Render the main interface with results."""
        results = st.session_state.omr_results
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🖼️ Image View", 
            "📊 Analysis", 
            "🎵 Musical Elements", 
            "⚙️ Corrections"
        ])
        
        with tab1:
            self._render_image_view(results)
        
        with tab2:
            self._render_analysis_view(results)
        
        with tab3:
            self._render_musical_elements_view(results)
        
        with tab4:
            self._render_corrections_view(results)
    
    def _render_image_view(self, results: Dict):
        """Render the image view with overlaid detections."""
        st.subheader("🖼️ Image with Detections")
        
        # Get the original and processed images
        original_image = results.get('original_image')
        symbols = results.get('symbols', [])
        
        if original_image is not None:
            # Create interactive plot
            fig = self._create_image_plot(original_image, symbols)
            
            # Display the plot
            selected_data = st.plotly_chart(
                fig, 
                use_container_width=True, 
                on_select="rerun",
                selection_mode="points"
            )
            
            # Handle symbol selection
            if selected_data and selected_data.selection and selected_data.selection.points:
                point = selected_data.selection.points[0]
                symbol_idx = point.get('customdata', [None])[0]
                if symbol_idx is not None:
                    self._handle_symbol_selection(symbol_idx, symbols)
        else:
            st.error("No image data available")
    
    def _render_analysis_view(self, results: Dict):
        """Render analysis and statistics view."""
        st.subheader("📊 Detection Analysis")
        
        symbols = results.get('symbols', [])
        staff_info = results.get('staff_info', [])
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Symbols", len(symbols))
        
        with col2:
            st.metric("Staves Detected", len(staff_info))
        
        with col3:
            if symbols:
                avg_conf = np.mean([s.confidence for s in symbols])
                st.metric("Avg Confidence", f"{avg_conf:.2f}")
        
        with col4:
            low_conf_count = len([s for s in symbols if s.confidence < st.session_state.conf_threshold])
            st.metric("Low Confidence", low_conf_count)
        
        # Symbol type distribution
        if symbols:
            symbol_types = {}
            for symbol in symbols:
                if symbol.class_name not in symbol_types:
                    symbol_types[symbol.class_name] = 0
                symbol_types[symbol.class_name] += 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Symbol Type Distribution")
                df_symbols = pd.DataFrame(
                    list(symbol_types.items()), 
                    columns=['Symbol Type', 'Count']
                )
                fig = px.bar(df_symbols, x='Symbol Type', y='Count')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Confidence Distribution")
                confidences = [s.confidence for s in symbols]
                fig = px.histogram(
                    x=confidences, 
                    nbins=20, 
                    title="Symbol Confidence Histogram"
                )
                fig.update_xaxes(title="Confidence Score")
                fig.update_yaxes(title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
        # Low confidence symbols table
        st.subheader("Low Confidence Symbols")
        low_conf_symbols = [
            s for s in symbols 
            if s.confidence < st.session_state.conf_threshold
        ]
        
        if low_conf_symbols:
            symbol_data = []
            for i, symbol in enumerate(low_conf_symbols):
                symbol_data.append({
                    'Index': i,
                    'Type': symbol.class_name,
                    'Confidence': f"{symbol.confidence:.3f}",
                    'Position': f"({symbol.center[0]:.0f}, {symbol.center[1]:.0f})",
                    'Pitch': symbol.pitch or 'N/A'
                })
            
            df = pd.DataFrame(symbol_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.success("No symbols below confidence threshold!")
    
    def _render_musical_elements_view(self, results: Dict):
        """Render the musical elements view."""
        st.subheader("🎵 Reconstructed Musical Elements")
        
        musical_elements = results.get('musical_elements')
        if not musical_elements:
            st.warning("No musical reconstruction data available")
            return
        
        staves = musical_elements.get('staves', [])
        
        for staff_idx, staff_data in enumerate(staves):
            with st.expander(f"Staff {staff_idx + 1}", expanded=True):
                
                # Staff information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    clef = staff_data.get('clef')
                    if clef:
                        st.write(f"**Clef:** {clef.clef_type}")
                        st.write(f"**Confidence:** {clef.confidence:.3f}")
                
                with col2:
                    time_sig = staff_data.get('time_signature')
                    if time_sig:
                        st.write(f"**Time Signature:** {time_sig.numerator}/{time_sig.denominator}")
                        st.write(f"**Confidence:** {time_sig.confidence:.3f}")
                
                with col3:
                    key_sig = staff_data.get('key_signature')
                    if key_sig:
                        st.write(f"**Key:** {key_sig.key}")
                        st.write(f"**Sharps/Flats:** {key_sig.sharps}")
                
                # Measures
                measures = staff_data.get('measures', [])
                if measures:
                    st.write(f"**Measures:** {len(measures)}")
                    
                    # Show notes for each measure
                    for measure_idx, measure in enumerate(measures):
                        notes = [e for e in measure.elements if isinstance(e, Note)]
                        if notes:
                            note_info = []
                            for note in notes:
                                note_str = f"{note.pitch}"
                                if note.accidental:
                                    note_str += f" ({note.accidental})"
                                note_str += f" [{note.duration}]"
                                note_info.append(note_str)
                            
                            st.write(f"Measure {measure_idx + 1}: {', '.join(note_info)}")
    
    def _render_corrections_view(self, results: Dict):
        """Render the corrections management view."""
        st.subheader("⚙️ Corrections Management")
        
        # Show current corrections
        if st.session_state.corrections:
            st.write("**Current Corrections:**")
            for symbol_idx, correction in st.session_state.corrections.items():
                st.write(f"Symbol {symbol_idx}: {correction}")
            
            if st.button("🗑️ Clear All Corrections"):
                st.session_state.corrections = {}
                st.rerun()
        else:
            st.info("No corrections made yet. Click on symbols in the Image View to edit them.")
        
        # Correction statistics
        st.subheader("📈 Correction Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Symbols Corrected", len(st.session_state.corrections))
        
        with col2:
            total_symbols = len(results.get('symbols', []))
            if total_symbols > 0:
                correction_rate = len(st.session_state.corrections) / total_symbols * 100
                st.metric("Correction Rate", f"{correction_rate:.1f}%")
    
    def _create_image_plot(self, image: np.ndarray, symbols: List[DetectedSymbol]) -> go.Figure:
        """Create an interactive plotly figure with the image and symbols."""
        # Convert image for display
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
        
        # Create base figure
        fig = go.Figure()
        
        # Add image
        fig.add_layout_image(
            dict(
                source=Image.fromarray(display_image),
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=image.shape[1],
                sizey=image.shape[0],
                sizing="stretch",
                opacity=0.8,
                layer="below"
            )
        )
        
        # Add symbols as scatter points
        if symbols:
            x_coords = [s.center[0] for s in symbols]
            y_coords = [image.shape[0] - s.center[1] for s in symbols]  # Flip Y for display
            
            # Color by confidence
            colors = [s.confidence for s in symbols]
            
            # Hover text
            hover_text = []
            for i, s in enumerate(symbols):
                text = f"Symbol {i}<br>Type: {s.class_name}<br>Confidence: {s.confidence:.3f}"
                if s.pitch:
                    text += f"<br>Pitch: {s.pitch}"
                hover_text.append(text)
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Confidence"),
                    line=dict(width=2, color='black')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                customdata=list(range(len(symbols))),
                name="Symbols"
            ))
            
            # Add bounding boxes if enabled
            if st.session_state.get('show_bboxes', True):
                for i, symbol in enumerate(symbols):
                    x, y, w, h = symbol.bbox
                    y_flipped = image.shape[0] - y - h  # Flip Y coordinate
                    
                    # Determine color based on confidence
                    color = 'red' if symbol.confidence < st.session_state.conf_threshold else 'green'
                    
                    fig.add_shape(
                        type="rect",
                        x0=x, y0=y_flipped,
                        x1=x + w, y1=y_flipped + h,
                        line=dict(color=color, width=2),
                        fillcolor='rgba(0,0,0,0)'
                    )
        
        # Update layout
        fig.update_layout(
            title="Click on symbols to edit them",
            xaxis=dict(
                range=[0, image.shape[1]],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=[0, image.shape[0]],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def _handle_symbol_selection(self, symbol_idx: int, symbols: List[DetectedSymbol]):
        """Handle when a user selects a symbol for editing."""
        if symbol_idx < len(symbols):
            symbol = symbols[symbol_idx]
            
            st.sidebar.markdown("---")
            st.sidebar.subheader(f"✏️ Edit Symbol {symbol_idx}")
            
            # Create form for editing
            with st.sidebar.form(f"edit_symbol_{symbol_idx}"):
                new_class = st.selectbox(
                    "Symbol Type",
                    options=[
                        'quarter_note', 'half_note', 'whole_note', 'eighth_note',
                        'quarter_rest', 'half_rest', 'whole_rest', 'eighth_rest',
                        'treble_clef', 'bass_clef', 'sharp', 'flat', 'natural'
                    ],
                    index=0 if symbol.class_name not in [
                        'quarter_note', 'half_note', 'whole_note', 'eighth_note',
                        'quarter_rest', 'half_rest', 'whole_rest', 'eighth_rest',
                        'treble_clef', 'bass_clef', 'sharp', 'flat', 'natural'
                    ] else [
                        'quarter_note', 'half_note', 'whole_note', 'eighth_note',
                        'quarter_rest', 'half_rest', 'whole_rest', 'eighth_rest',
                        'treble_clef', 'bass_clef', 'sharp', 'flat', 'natural'
                    ].index(symbol.class_name)
                )
                
                new_confidence = st.slider(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=symbol.confidence,
                    step=0.01
                )
                
                new_pitch = st.text_input(
                    "Pitch (for notes)",
                    value=symbol.pitch or ""
                )
                
                if st.form_submit_button("💾 Save Changes"):
                    # Store the correction
                    st.session_state.corrections[symbol_idx] = {
                        'original_class': symbol.class_name,
                        'new_class': new_class,
                        'original_confidence': symbol.confidence,
                        'new_confidence': new_confidence,
                        'original_pitch': symbol.pitch,
                        'new_pitch': new_pitch if new_pitch else None
                    }
                    
                    # Apply the correction to the symbol
                    symbol.class_name = new_class
                    symbol.confidence = new_confidence
                    symbol.pitch = new_pitch if new_pitch else None
                    
                    st.success(f"Symbol {symbol_idx} updated!")
                    st.rerun()
    
    def _process_uploaded_image(self, uploaded_file):
        """Process an uploaded image file."""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize pipeline and process
            with st.spinner("Processing image with OMR pipeline..."):
                if self.pipeline is None:
                    self.pipeline = OMRPipeline()
                
                results = self.pipeline.process_image(temp_path)
                st.session_state.omr_results = results
                st.session_state.current_image = temp_path
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            st.success("Image processed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    def _load_existing_results(self, results_file):
        """Load existing OMR results from JSON file."""
        try:
            results_data = json.load(results_file)
            # Convert JSON back to OMR objects if needed
            # This would require deserialization logic
            st.session_state.omr_results = results_data
            st.success("Results loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
    
    def _export_musicxml(self):
        """Export corrected results as MusicXML."""
        try:
            if self.pipeline and st.session_state.omr_results:
                output_path = "corrected_output.mxl"
                self.pipeline.save_musicxml(st.session_state.omr_results, output_path)
                
                # Provide download link
                with open(output_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 Download MusicXML",
                        data=f.read(),
                        file_name="corrected_music.mxl",
                        mime="application/xml"
                    )
                    
        except Exception as e:
            st.error(f"Error exporting MusicXML: {str(e)}")
    
    def _export_corrected_json(self):
        """Export corrected results as JSON."""
        try:
            if st.session_state.omr_results:
                # Include corrections in the export
                export_data = st.session_state.omr_results.copy()
                export_data['manual_corrections'] = st.session_state.corrections
                
                json_str = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Corrected JSON",
                    data=json_str,
                    file_name="corrected_omr_results.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"Error exporting JSON: {str(e)}")


def main():
    """Main function to run the Streamlit app."""
    ui = OMRCorrectionUI()
    ui.run()


if __name__ == "__main__":
    main()
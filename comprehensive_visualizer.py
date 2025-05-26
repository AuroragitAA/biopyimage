# comprehensive_visualizer.py
import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class ComprehensiveVisualizer:
    """Generate comprehensive visualizations for Wolffia analysis"""
    
    def __init__(self):
        self.figure_dpi = 150
        self.color_palette = sns.color_palette("husl", 8)
        
    def create_all_visualizations(self, analysis_result):
        """Create all visualization types"""
        visualizations = {}
        
        # 1. Biomass visualizations
        if 'biomass_analysis' in analysis_result:
            visualizations['biomass_charts'] = self.create_biomass_visualizations(
                analysis_result['biomass_analysis']
            )
        
        # 2. Spectral/Wavelength visualizations
        if 'spectral_analysis' in analysis_result:
            visualizations['spectral_charts'] = self.create_spectral_visualizations(
                analysis_result['spectral_analysis']
            )
        
        # 3. Cell similarity visualizations
        if 'similarity_analysis' in analysis_result:
            visualizations['similarity_charts'] = self.create_similarity_visualizations(
                analysis_result['similarity_analysis']
            )
        
        # 4. Temporal tracking visualizations
        if 'temporal_analysis' in analysis_result:
            visualizations['temporal_charts'] = self.create_temporal_visualizations(
                analysis_result['temporal_analysis']
            )
        
        # 5. Comprehensive dashboard
        visualizations['dashboard'] = self.create_analysis_dashboard(analysis_result)
        
        return visualizations
    
    def create_biomass_visualizations(self, biomass_data):
        """Create biomass-related visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Biomass Estimation Methods', 'Biomass Confidence Intervals',
                          'Area vs Biomass Relationship', 'Biomass Components'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # 1. Biomass estimation comparison
        methods = ['Area-based', 'Chlorophyll-based', 'Allometric']
        values = [
            biomass_data['area_based']['fresh_biomass_g'],
            biomass_data['chlorophyll_based']['estimated_biomass_g'],
            biomass_data['allometric']['estimated_biomass_g']
        ]
        
        fig.add_trace(
            go.Bar(x=methods, y=values, name='Biomass (g)',
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']),
            row=1, col=1
        )
        
        # 2. Confidence intervals
        combined = biomass_data['combined_estimate']
        fig.add_trace(
            go.Scatter(
                x=['Combined Estimate'],
                y=[combined['fresh_biomass_g']],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[combined['confidence_interval'][1] - combined['fresh_biomass_g']],
                    arrayminus=[combined['fresh_biomass_g'] - combined['confidence_interval'][0]]
                ),
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Biomass with CI'
            ),
            row=1, col=2
        )
        
        # 3. Area-Biomass relationship
        areas = np.linspace(100, 5000, 50)
        biomass_est = 0.0012 * (areas ** 1.15) * 1e-6
        
        fig.add_trace(
            go.Scatter(x=areas, y=biomass_est, mode='lines',
                      name='Allometric Model', line=dict(color='green')),
            row=2, col=1
        )
        
        # 4. Biomass components pie chart
        fig.add_trace(
            go.Pie(
                labels=['Fresh Biomass', 'Water Content'],
                values=[combined['dry_biomass_g'], 
                       combined['fresh_biomass_g'] - combined['dry_biomass_g']],
                hole=0.3
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Biomass Analysis Dashboard")
        
        return self._fig_to_base64(fig)
    
    def create_spectral_visualizations(self, spectral_data):
        """Create wavelength/spectral visualizations"""
        if not spectral_data.get('cell_spectral_data'):
            return None
        
        df = pd.DataFrame(spectral_data['cell_spectral_data'])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spectral Intensity Distribution', 'Chlorophyll Content by Cell',
                          'Wavelength Intensity Heatmap', 'Vegetation Indices'),
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Box plots of spectral intensities
        wavelengths = ['green_550nm', 'red_665nm', 'blue_465nm']
        for i, wl in enumerate(wavelengths):
            if wl in spectral_data['wavelength_distribution']:
                fig.add_trace(
                    go.Box(y=spectral_data['wavelength_distribution'][wl],
                          name=wl, marker_color=self.color_palette[i]),
                    row=1, col=1
                )
        
        # 2. Chlorophyll content bar chart
        fig.add_trace(
            go.Bar(x=df['cell_id'], y=df['total_chlorophyll'],
                  name='Total Chlorophyll', marker_color='green'),
            row=1, col=2
        )
        
        # 3. Scatter plot of green intensity vs chlorophyll
        fig.add_trace(
            go.Scatter(x=df['green_intensity_550nm'], y=df['total_chlorophyll'],
                      mode='markers', marker=dict(size=8, color=df['cell_id'],
                      colorscale='Viridis'), name='Green-Chlorophyll Correlation'),
            row=2, col=1
        )
        
        # 4. Vegetation indices
        fig.add_trace(
            go.Scatter(x=df['cell_id'], y=df['vegetation_index'],
                      mode='lines+markers', name='Vegetation Index',
                      line=dict(color='darkgreen')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Spectral Analysis Dashboard")
        
        return self._fig_to_base64(fig)
    
    def create_similarity_visualizations(self, similarity_data):
        """Create cell similarity visualizations"""
        import plotly.express as px
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Similarity Matrix Heatmap', 'Cluster Distribution',
                          'Similar Cell Pairs', 'Cluster Characteristics')
        )
        
        # 1. Similarity matrix heatmap
        if similarity_data.get('similarity_matrix'):
            matrix = np.array(similarity_data['similarity_matrix'])
            heatmap = go.Heatmap(z=matrix, colorscale='Viridis')
            fig.add_trace(heatmap, row=1, col=1)
        
        # 2. Cluster distribution
        if 'hierarchical' in similarity_data.get('similarity_clusters', {}):
            labels = similarity_data['similarity_clusters']['hierarchical']
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            fig.add_trace(
                go.Bar(x=[f'Cluster {l}' for l in unique_labels], y=counts,
                      marker_color=self.color_palette[:len(unique_labels)]),
                row=1, col=2
            )
        
        # 3. Similar cell pairs network
        if similarity_data.get('similar_cell_groups'):
            # Create a simple network visualization
            pairs = similarity_data['similar_cell_groups'][:10]  # Top 10 pairs
            x_pos = []
            y_pos = []
            for i, pair in enumerate(pairs):
                x_pos.extend([i, i])
                y_pos.extend([0, 1])
            
            fig.add_trace(
                go.Scatter(x=x_pos, y=y_pos, mode='markers+text',
                          text=[f"Cell {p['cell_1']}" if i%2==0 else f"Cell {p['cell_2']}" 
                                for i, p in enumerate(pairs) for _ in range(2)],
                          marker=dict(size=10)),
                row=2, col=1
            )
        
        # 4. Cluster characteristics
        if 'hierarchical' in similarity_data.get('cluster_statistics', {}):
            cluster_stats = similarity_data['cluster_statistics']['hierarchical']
            if cluster_stats:
                cluster_ids = [stat['cluster_id'] for stat in cluster_stats]
                avg_areas = [stat['avg_area'] for stat in cluster_stats]
                
                fig.add_trace(
                    go.Scatter(x=cluster_ids, y=avg_areas, mode='markers',
                              marker=dict(size=[stat['cell_count']*10 for stat in cluster_stats],
                                        color=cluster_ids, colorscale='Viridis'),
                              name='Avg Area by Cluster'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Cell Similarity Analysis")
        
        return self._fig_to_base64(fig)
    
    def create_temporal_visualizations(self, temporal_data):
        """Create temporal tracking visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population Growth Curve', 'Individual Cell Tracks',
                          'Growth Rate Distribution', 'Growth Phases'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Population growth curve
        if 'population_dynamics' in temporal_data:
            pop_dyn = temporal_data['population_dynamics']
            fig.add_trace(
                go.Scatter(x=pop_dyn['time_points'], y=pop_dyn['total_cells'],
                          mode='lines+markers', name='Total Cells',
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            # Add growth phases if available
            if 'growth_phases' in pop_dyn:
                for phase in pop_dyn['growth_phases']:
                    fig.add_vrect(x0=phase['start'], x1=phase['end'],
                                fillcolor={'exponential': 'green', 'stationary': 'yellow',
                                         'decline': 'red'}.get(phase['phase'], 'gray'),
                                opacity=0.2, line_width=0, row=1, col=1)
        
        # 2. Individual cell tracks
        if 'growth_curves' in temporal_data:
            for i, (track_id, curve) in enumerate(list(temporal_data['growth_curves'].items())[:10]):
                fig.add_trace(
                    go.Scatter(x=curve['time_points'], y=curve['areas'],
                              mode='lines+markers', name=track_id,
                              line=dict(color=self.color_palette[i % 8])),
                    row=1, col=2
                )
        
        # 3. Growth rate histogram
        if 'growth_curves' in temporal_data:
            growth_rates = [curve['growth_rate'] for curve in temporal_data['growth_curves'].values()]
            fig.add_trace(
                go.Histogram(x=growth_rates, nbinsx=20, name='Growth Rate Distribution'),
                row=2, col=1
            )
        
        # 4. Division events over time
        if 'division_events' in temporal_data and temporal_data['division_events']:
            div_times = [event['time_point'] for event in temporal_data['division_events']]
            div_counts = pd.Series(div_times).value_counts().sort_index()
            
            fig.add_trace(
                go.Scatter(x=div_counts.index, y=div_counts.values,
                          mode='markers+lines', name='Division Events',
                          marker=dict(size=10, color='red')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Temporal Analysis Dashboard")
        
        return self._fig_to_base64(fig)
    
    def create_analysis_dashboard(self, analysis_result):
        """Create comprehensive analysis dashboard"""
        # Create a multi-panel dashboard with key metrics
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Cell Count', 'Size Distribution', 'Health Score Distribution',
                          'Biomass Estimate', 'Chlorophyll Content', 'Similarity Clusters',
                          'Temporal Tracking', 'Spectral Properties', 'Quality Metrics'),
            specs=[[{"type": "indicator"}, {"type": "histogram"}, {"type": "box"}],
                   [{"type": "indicator"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 1. Cell count indicator
        total_cells = analysis_result.get('total_cells', 0)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=total_cells,
                title={"text": "Total Cells"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # 2. Size distribution
        if analysis_result.get('cell_data'):
            df = pd.DataFrame(analysis_result['cell_data'])
            fig.add_trace(
                go.Histogram(x=df['area'], nbinsx=30, name='Cell Size Distribution'),
                row=1, col=2
            )
            
            # 3. Health score distribution
            if 'health_score' in df.columns:
                fig.add_trace(
                    go.Box(y=df['health_score'], name='Health Scores'),
                    row=1, col=3
                )
        
        # 4. Biomass indicator
        if 'biomass_analysis' in analysis_result:
            biomass_g = analysis_result['biomass_analysis']['combined_estimate']['fresh_biomass_g']
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=biomass_g,
                    number={'suffix': " g"},
                    title={"text": "Estimated Biomass"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=1
            )
        
        # Add more visualizations based on available data...
        
        fig.update_layout(height=1200, title_text="Comprehensive Wolffia Analysis Dashboard")
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Convert plotly figure to base64 string"""
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        img_base64 = base64.b64encode(img_bytes).decode()
        return img_base64
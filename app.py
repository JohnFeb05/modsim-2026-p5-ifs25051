# ============================================================================
# APLIKASI STREAMLIT - ESTIMASI WAKTU PEMBANGUNAN GEDUNG FITE
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Simulasi Monte Carlo - Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================

class ProjectStage:
    """Kelas untuk memodelkan tahapan proyek"""
    
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []
        
    def sample_duration(self, n_simulations, risk_multiplier=1.0):
        """Sampling durasi dengan mempertimbangkan distribusi dan faktor risiko"""
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )
        
        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                probability = risk_params['probability']
                impact = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration
                )
            elif risk_params['type'] == 'continuous':
                mean = risk_params['mean']
                std = risk_params['std']
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)
        
        return base_duration * risk_multiplier


class MonteCarloProjectSimulation:
    """Kelas untuk menjalankan simulasi Monte Carlo"""
    
    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self.initialize_stages()
        
    def initialize_stages(self):
        """Inisialisasi objek tahapan dari konfigurasi"""
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = ProjectStage(
                name=stage_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )
    
    def run_simulation(self):
        """Menjalankan simulasi Monte Carlo lengkap"""
        results = pd.DataFrame(index=range(self.num_simulations))
        
        for stage_name, stage in self.stages.items():
            results[stage_name] = stage.sample_duration(self.num_simulations)
        
        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))
        
        for stage_name in self.stages.keys():
            deps = self.stages[stage_name].dependencies
            
            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)
            
            end_times[stage_name] = start_times[stage_name] + results[stage_name]
        
        results['Total_Duration'] = end_times.max(axis=1)
        
        for stage_name in self.stages.keys():
            results[f'{stage_name}_Finish'] = end_times[stage_name]
            results[f'{stage_name}_Start'] = start_times[stage_name]
        
        self.simulation_results = results
        return results
    
    def calculate_critical_path_probability(self):
        """Menghitung probabilitas setiap tahapan berada di critical path"""
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        critical_path_probs = {}
        total_duration = self.simulation_results['Total_Duration']
        
        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{stage_name}_Finish']
            correlation = self.simulation_results[stage_name].corr(total_duration)
            is_critical = (stage_finish + 0.1) >= total_duration
            prob_critical = np.mean(is_critical)
            
            critical_path_probs[stage_name] = {
                'probability': prob_critical,
                'correlation': correlation,
                'avg_duration': self.simulation_results[stage_name].mean()
            }
        
        return pd.DataFrame(critical_path_probs).T
    
    def analyze_risk_contribution(self):
        """Analisis kontribusi risiko terhadap variabilitas total durasi"""
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        
        total_var = self.simulation_results['Total_Duration'].var()
        contributions = {}
        
        for stage_name in self.stages.keys():
            stage_var = self.simulation_results[stage_name].var()
            stage_covar = self.simulation_results[stage_name].cov(
                self.simulation_results['Total_Duration']
            )
            contribution = (stage_covar / total_var) * 100
            
            contributions[stage_name] = {
                'variance': stage_var,
                'contribution_percent': contribution,
                'std_dev': np.sqrt(stage_var)
            }
        
        return pd.DataFrame(contributions).T


# ============================================================================
# 3. FUNGSI VISUALISASI PLOTLY
# ============================================================================

def create_distribution_plot(results):
    """Membuat plot distribusi durasi total proyek"""
    total_duration = results['Total_Duration']
    mean_duration = total_duration.mean()
    median_duration = np.median(total_duration)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=total_duration,
        nbinsx=50,
        name='Distribusi Durasi',
        marker_color='skyblue',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_duration:.1f}")
    fig.add_vline(x=median_duration, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_duration:.1f}")
    
    ci_80 = np.percentile(total_duration, [10, 90])
    ci_95 = np.percentile(total_duration, [2.5, 97.5])
    
    fig.add_vrect(x0=ci_80[0], x1=ci_80[1], fillcolor="yellow", opacity=0.2,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci_95[0], x1=ci_95[1], fillcolor="orange", opacity=0.1,
                  annotation_text="95% CI", line_width=0)
    
    fig.update_layout(
        title='Distribusi Durasi Total Proyek',
        xaxis_title='Durasi Total Proyek (Bulan)',
        yaxis_title='Densitas Probabilitas',
        showlegend=True,
        height=500
    )
    
    return fig, {
        'mean': mean_duration,
        'median': median_duration,
        'std': total_duration.std(),
        'min': total_duration.min(),
        'max': total_duration.max(),
        'ci_80': ci_80,
        'ci_95': ci_95
    }


def create_completion_probability_plot(results):
    """Membuat plot probabilitas penyelesaian proyek"""
    deadlines = np.arange(12, 30, 1)
    completion_probs = []
    
    for deadline in deadlines:
        prob = np.mean(results['Total_Duration'] <= deadline)
        completion_probs.append(prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=deadlines,
        y=completion_probs,
        mode='lines',
        name='Probabilitas Selesai',
        line=dict(color='darkblue', width=3),
        fill='tozeroy',
        fillcolor='rgba(173, 216, 230, 0.3)'
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="50%", annotation_position="right")
    fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                  annotation_text="80%", annotation_position="right")
    fig.add_hline(y=0.95, line_dash="dash", line_color="blue",
                  annotation_text="95%", annotation_position="right")
    
    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[12, 30],
        height=500
    )
    
    return fig


def create_critical_path_plot(critical_analysis):
    """Membuat plot analisis critical path"""
    critical_analysis = critical_analysis.sort_values('probability', ascending=True)
    
    fig = go.Figure()
    
    colors = ['red' if prob > 0.7 else 'lightcoral' for prob in critical_analysis['probability']]
    
    fig.add_trace(go.Bar(
        y=[stage.replace('_', ' ') for stage in critical_analysis.index],
        x=critical_analysis['probability'],
        orientation='h',
        marker_color=colors,
        text=[f'{prob:.1%}' for prob in critical_analysis['probability']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Analisis Critical Path per Tahapan',
        xaxis_title='Probabilitas Menjadi Critical Path',
        xaxis_range=[0, 1.0],
        height=500
    )
    
    return fig


def create_stage_boxplot(results, stages):
    """Membuat boxplot distribusi durasi per tahapan"""
    stage_names = list(stages.keys())
    stage_data = [results[stage] for stage in stage_names]
    
    fig = go.Figure()
    
    for i, (stage, data) in enumerate(zip(stage_names, stage_data)):
        fig.add_trace(go.Box(
            y=data,
            name=stage.replace('_', '\n'),
            boxmean='sd',
            marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title='Distribusi Durasi per Tahapan',
        yaxis_title='Durasi (Bulan)',
        height=500,
        showlegend=False
    )
    
    return fig


def create_risk_contribution_plot(risk_contrib):
    """Membuat plot kontribusi risiko per tahapan"""
    risk_contrib = risk_contrib.sort_values('contribution_percent', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[name.replace('_', '\n') for name in risk_contrib.index],
        y=risk_contrib['contribution_percent'],
        marker_color=px.colors.qualitative.Set3,
        text=[f'{contrib:.1f}%' for contrib in risk_contrib['contribution_percent']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Kontribusi Risiko per Tahapan',
        yaxis_title='Kontribusi terhadap Variabilitas (%)',
        height=400
    )
    
    return fig


# ============================================================================
# 4. FUNGSI UTAMA STREAMLIT
# ============================================================================

def main():
    # Header aplikasi
    st.markdown('<h1 class="main-header">🏗️ Simulasi Monte Carlo - Gedung FITE</h1>', unsafe_allow_html=True)
    
    # Deskripsi
    st.markdown("""
    <div class="info-box">
    Estimasi waktu penyelesaian proyek pembangunan gedung FITE 5 lantai dengan fasilitas lengkap.
    Aplikasi ini menggunakan simulasi Monte Carlo untuk memodelkan ketidakpastian.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar untuk konfigurasi
    st.sidebar.markdown('<h2>⚙️ Konfigurasi Simulasi</h2>', unsafe_allow_html=True)
    
    # Slider untuk jumlah simulasi (dikurangi untuk menghindari timeout)
    num_simulations = st.sidebar.slider(
        'Jumlah Iterasi Simulasi:',
        min_value=1000,
        max_value=10000,  # Dikurangi dari 50000 untuk menghindari timeout
        value=5000,
        step=1000,
        help='Semakin banyak iterasi, semakin akurat hasilnya'
    )
    
    # Konfigurasi default untuk Gedung FITE
    default_config = {
        "Persiapan_Lahan": {
            "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
            "risk_factors": {
                "kondisi_tanah": {"type": "discrete", "probability": 0.3, "impact": 0.4}
            }
        },
        "Pondasi_Dan_Struktur_Dasar": {
            "base_params": {"optimistic": 4, "most_likely": 6, "pessimistic": 9},
            "risk_factors": {
                "cuaca_buruk": {"type": "discrete", "probability": 0.4, "impact": 0.25}
            },
            "dependencies": ["Persiapan_Lahan"]
        },
        "Struktur_Bangunan_5_Lantai": {
            "base_params": {"optimistic": 6, "most_likely": 9, "pessimistic": 14},
            "risk_factors": {
                "produktivitas_pekerja": {"type": "continuous", "mean": 1.0, "std": 0.25}
            },
            "dependencies": ["Pondasi_Dan_Struktur_Dasar"]
        },
        "Instalasi_Listrik_Dan_Mekanikal": {
            "base_params": {"optimistic": 3, "most_likely": 5, "pessimistic": 8},
            "risk_factors": {
                "keterlambatan_peralatan": {"type": "discrete", "probability": 0.3, "impact": 0.35}
            },
            "dependencies": ["Struktur_Bangunan_5_Lantai"]
        },
        "Pembangunan_Laboratorium_Khusus": {
            "base_params": {"optimistic": 4, "most_likely": 6, "pessimistic": 10},
            "risk_factors": {
                "perubahan_desain_lab": {"type": "discrete", "probability": 0.4, "impact": 0.45}
            },
            "dependencies": ["Instalasi_Listrik_Dan_Mekanikal"]
        },
        "Finishing_Dan_Interior": {
            "base_params": {"optimistic": 3, "most_likely": 4, "pessimistic": 7},
            "risk_factors": {
                "ketersediaan_material": {"type": "discrete", "probability": 0.2, "impact": 0.25}
            },
            "dependencies": ["Pembangunan_Laboratorium_Khusus"]
        },
        "Pengujian_Sistem_Dan_Komisining": {
            "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
            "risk_factors": {
                "masalah_sistem": {"type": "discrete", "probability": 0.3, "impact": 0.5}
            },
            "dependencies": ["Finishing_Dan_Interior"]
        },
        "Serah_Terima_Dan_Operasional": {
            "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 4},
            "risk_factors": {
                "dokumentasi": {"type": "discrete", "probability": 0.2, "impact": 0.3}
            },
            "dependencies": ["Pengujian_Sistem_Dan_Komisining"]
        }
    }
    
    # Menampilkan konfigurasi tahapan di sidebar
    for stage_name, config in default_config.items():
        with st.sidebar.expander(f"⚙️ {stage_name.replace('_', ' ')}", expanded=False):
            optimistic = st.number_input(
                f"Optimistic",
                min_value=1,
                max_value=100,
                value=config['base_params']['optimistic'],
                key=f"opt_{stage_name}"
            )
            
            most_likely = st.number_input(
                f"Most Likely",
                min_value=1,
                max_value=100,
                value=config['base_params']['most_likely'],
                key=f"ml_{stage_name}"
            )
            
            pessimistic = st.number_input(
                f"Pessimistic",
                min_value=1,
                max_value=100,
                value=config['base_params']['pessimistic'],
                key=f"pes_{stage_name}"
            )
            
            default_config[stage_name]['base_params'] = {
                'optimistic': optimistic,
                'most_likely': most_likely,
                'pessimistic': pessimistic
            }
    
    # Tombol untuk menjalankan
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Inisialisasi session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    
    # Jalankan simulasi ketika tombol ditekan
    if run_simulation:
        with st.spinner('Menjalankan simulasi Monte Carlo... Harap tunggu...'):
            try:
                simulator = MonteCarloProjectSimulation(
                    stages_config=default_config,
                    num_simulations=num_simulations
                )
                
                results = simulator.run_simulation()
                
                st.session_state.simulation_results = results
                st.session_state.simulator = simulator
                
                st.success(f'Simulasi selesai! {num_simulations:,} iterasi berhasil dijalankan.')
            except Exception as e:
                st.error(f'Error saat simulasi: {str(e)}')
                st.stop()
    
    # Tampilkan hasil jika simulasi sudah dijalankan
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        simulator = st.session_state.simulator
        
        # BAGIAN 1: STATISTIK UTAMA
        st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)
        
        total_duration = results['Total_Duration']
        mean_duration = total_duration.mean()
        median_duration = np.median(total_duration)
        ci_80 = np.percentile(total_duration, [10, 90])
        ci_95 = np.percentile(total_duration, [2.5, 97.5])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{mean_duration:.1f}</h3>
                <p>Rata-rata Durasi (Bulan)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{median_duration:.1f}</h3>
                <p>Median Durasi (Bulan)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{ci_80[0]:.1f} - {ci_80[1]:.1f}</h3>
                <p>80% Confidence Interval</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{ci_95[0]:.1f} - {ci_95[1]:.1f}</h3>
                <p>95% Confidence Interval</p>
            </div>
            """, unsafe_allow_html=True)
        
        # BAGIAN 2: VISUALISASI
        st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Distribusi Durasi",
            "🎯 Probabilitas Penyelesaian",
            "🔍 Analisis Tahapan",
            "📊 Analisis Risiko"
        ])
        
        with tab1:
            fig_dist, stats = create_distribution_plot(results)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab2:
            fig_prob = create_completion_probability_plot(results)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                critical_analysis = simulator.calculate_critical_path_probability()
                fig_critical = create_critical_path_plot(critical_analysis)
                st.plotly_chart(fig_critical, use_container_width=True)
            with col2:
                fig_boxplot = create_stage_boxplot(results, simulator.stages)
                st.plotly_chart(fig_boxplot, use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                risk_contrib = simulator.analyze_risk_contribution()
                fig_risk = create_risk_contribution_plot(risk_contrib)
                st.plotly_chart(fig_risk, use_container_width=True)
            with col2:
                st.info("Heatmap korelasi dapat ditambahkan di sini")
        
        # BAGIAN 3: REKOMENDASI
        st.markdown('<h2 class="sub-header">🎯 Rekomendasi</h2>', unsafe_allow_html=True)
        
        safety_buffer = np.percentile(total_duration, 80) - mean_duration
        contingency_reserve = np.percentile(total_duration, 95) - mean_duration
        
        st.markdown(f"""
        <div class="info-box">
            <h4>🏗️ Manajemen Risiko:</h4>
            • <b>Safety Buffer</b> (80% confidence): <b>{safety_buffer:.1f} bulan</b><br>
            • <b>Contingency Reserve</b> (95% confidence): <b>{contingency_reserve:.1f} bulan</b><br>
            • <b>Estimasi direkomendasikan:</b> {mean_duration:.1f} + {safety_buffer:.1f} = <b>{mean_duration + safety_buffer:.1f} bulan</b>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background-color: #f8f9fa; border-radius: 10px;">
            <h3>🚀 Siap untuk memulai simulasi?</h3>
            <p>Atur parameter di sidebar kiri, lalu klik tombol <b>"Run Simulation"</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><b>Simulasi Monte Carlo - Gedung FITE</b></p>
    <p>⚠️ Hasil simulasi merupakan estimasi probabilistik</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
# visualization_generator.py
import os
import nibabel as nib
import numpy as np
from skimage import measure, morphology
import plotly.graph_objects as go


def generate_visualization_html(folder_path):
    """
    Generates a 3D tumor + brain visualization HTML
    with home + download + toggle controls.
    Saves as visualization.html inside that folder.
    """
    try:
        brain_path = os.path.join(folder_path, "overlaid.nii")
        tumor_path = os.path.join(folder_path, "pred_mask.nii")

        if not (os.path.exists(brain_path) and os.path.exists(tumor_path)):
            raise FileNotFoundError("Missing overlaid.nii or pred_mask.nii")

        brain = nib.load(brain_path).get_fdata()
        tumor = nib.load(tumor_path).get_fdata()

        if brain.ndim == 4:
            brain = brain[..., 0]

        tumor_mask = (tumor > 0).astype(np.uint8)
        tumor_mask = morphology.binary_closing(tumor_mask, morphology.ball(1))

        verts_b, faces_b, _, _ = measure.marching_cubes((brain > brain.mean()), 0)
        verts_t, faces_t, _, _ = measure.marching_cubes(tumor_mask, 0)

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=verts_b[:, 0], y=verts_b[:, 1], z=verts_b[:, 2],
            i=faces_b[:, 0], j=faces_b[:, 1], k=faces_b[:, 2],
            color='gray', opacity=0.25, name='Brain'
        ))
        fig.add_trace(go.Mesh3d(
            x=verts_t[:, 0], y=verts_t[:, 1], z=verts_t[:, 2],
            i=faces_t[:, 0], j=faces_t[:, 1], k=faces_t[:, 2],
            color='red', opacity=0.85, name='Tumor'
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            title="3D Brain & Tumor Visualization",
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # --- Modern, integrated UI ---
        ui_html = """
        <style>
            body {
                margin: 0;
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #f7f9fb, #e3f2fd);
                overflow: hidden;
            }

            #controlBar {
                position: fixed;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                border-radius: 40px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px 25px;
                gap: 20px;
                z-index: 1000;
                transition: all 0.3s ease;
            }

            .btn {
                background: linear-gradient(135deg, #1976d2, #42a5f5);
                color: white;
                border: none;
                padding: 8px 18px;
                border-radius: 25px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 14px;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            }

            .toggle-group {
                display: flex;
                align-items: center;
                gap: 6px;
                font-weight: 600;
                color: #333;
            }

            .switch {
                position: relative;
                display: inline-block;
                width: 50px;
                height: 26px;
            }

            .switch input { display: none; }

            .slider {
                position: absolute;
                cursor: pointer;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: #bbb;
                transition: 0.2s;
                border-radius: 26px;
            }

            .slider:before {
                position: absolute;
                content: "";
                height: 20px; width: 20px;
                left: 3px; bottom: 3px;
                background-color: white;
                transition: 0.2s;
                border-radius: 50%;
            }

            input:checked + .slider {
                background-color: #4CAF50;
            }

            input:checked + .slider:before {
                transform: translateX(24px);
            }

            #plot-container {
                position: absolute;
                top: 0; left: 0;
                width: 100%; height: 100%;
            }
        </style>

        <div id="controlBar">
            <button class="btn" onclick="window.location.href='/'">🏠 Home</button>

            <div class="toggle-group">
                <span>🧠 Brain</span>
                <label class="switch">
                    <input type="checkbox" id="brainToggle" checked>
                    <span class="slider"></span>
                </label>
            </div>

            <div class="toggle-group">
                <span>🎯 Tumor</span>
                <label class="switch">
                    <input type="checkbox" id="tumorToggle" checked>
                    <span class="slider"></span>
                </label>
            </div>

            <button class="btn" id="downloadBtn">📥 Download PNG</button>
        </div>

        <script>
            document.addEventListener("DOMContentLoaded", () => {
                const plot = document.querySelector('.js-plotly-plot');
                const brainToggle = document.getElementById('brainToggle');
                const tumorToggle = document.getElementById('tumorToggle');
                const downloadBtn = document.getElementById('downloadBtn');

                function setVisibility(traceIdx, visible) {
                    Plotly.restyle(plot, { visible: visible ? true : "legendonly" }, [traceIdx]);
                }

                brainToggle.addEventListener("change", e => setVisibility(0, e.target.checked));
                tumorToggle.addEventListener("change", e => setVisibility(1, e.target.checked));

                downloadBtn.addEventListener("click", () => {
                    Plotly.downloadImage(plot, {
                        format: 'png',
                        filename: 'tumor_visualization',
                        height: 800,
                        width: 1000,
                        scale: 2
                    });
                });
            });
        </script>
        """

        out_path = os.path.join(folder_path, "visualization.html")
        fig.write_html(out_path, include_plotlyjs='cdn', full_html=False)

        # Wrap in container and UI
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("<div id='plot-container'>")
            fig.write_html(f, include_plotlyjs='cdn', full_html=False)
            f.write("</div>")
            f.write(ui_html)

        return out_path

    except Exception as e:
        raise RuntimeError(f"Visualization generation failed: {e}")

import os
import nibabel as nib
import numpy as np
from skimage import measure, morphology
import plotly.graph_objects as go

def generate_combined_visualization_html(patient_folder):
    """
    Generates a combined 3D visualization for all days in a patient's folder.
    Includes UI for Home, Download, Play/Pause animation, toggles,
    and a glowing, interactive color legend that highlights the current day.
    """
    import os
    import nibabel as nib
    import numpy as np
    from skimage import measure, morphology
    import plotly.graph_objects as go

    try:
        if not os.path.isdir(patient_folder):
            raise FileNotFoundError(f"{patient_folder} not found")

        day_dirs = sorted(
            [d for d in os.listdir(patient_folder) if d.lower().startswith("day")],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )
        if not day_dirs:
            raise FileNotFoundError("No Day folders found for this patient")

        colors = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
        ]

        fig = go.Figure()
        has_any = False
        color_map = {}

        # Optional: faint gray brain outline from first day
        first_day = os.path.join(patient_folder, day_dirs[0])
        brain_path = os.path.join(first_day, "overlaid.nii")
        if os.path.exists(brain_path):
            brain = nib.load(brain_path).get_fdata()
            if brain.ndim == 4:
                brain = brain[..., 0]
            verts_b, faces_b, _, _ = measure.marching_cubes((brain > brain.mean()), 0)
            fig.add_trace(go.Mesh3d(
                x=verts_b[:, 0], y=verts_b[:, 1], z=verts_b[:, 2],
                i=faces_b[:, 0], j=faces_b[:, 1], k=faces_b[:, 2],
                color='gray', opacity=0.15, name='Brain'
            ))

        # Tumor meshes for each day
        for idx, day in enumerate(day_dirs):
            day_path = os.path.join(patient_folder, day)
            tumor_path = os.path.join(day_path, "pred_mask.nii")
            if not os.path.exists(tumor_path):
                continue

            tumor = nib.load(tumor_path).get_fdata()
            tumor_mask = (tumor > 0).astype(np.uint8)
            if not tumor_mask.any():
                continue

            tumor_mask = morphology.binary_closing(tumor_mask, morphology.ball(1))
            verts, faces, _, _ = measure.marching_cubes(tumor_mask, 0)

            color = colors[idx % len(colors)]
            color_map[day] = color
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=0.6, name=day
            ))
            has_any = True

        if not has_any:
            raise RuntimeError("No valid tumor data found for combined view")

        fig.update_layout(
            title="Combined 3D Tumor Visualization (All Days)",
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        out_path = os.path.join(patient_folder, "combined_visualization.html")
        fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)

        # Legend HTML items
        legend_items = "".join(
            [f"<div class='legend-item' id='legend-{i}'><div class='color-dot' style='background:{color};'></div><span>{day}</span></div>"
             for i, (day, color) in enumerate(color_map.items())]
        )

        # === UI Bar + Play + Glowing Legend ===
        toggle_html = f"""
        <style>
            body {{
                margin: 0;
                background: linear-gradient(135deg, #f7f9fb, #e3f2fd);
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            #controlBar {{
                position: fixed;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                border-radius: 40px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px 25px;
                gap: 15px;
                z-index: 9999;
            }}
            .btn {{
                background: linear-gradient(135deg, #1976d2, #42a5f5);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 25px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 14px;
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            }}
            .toggle-group {{
                display: flex;
                align-items: center;
                gap: 5px;
                font-weight: 600;
                color: #333;
            }}
            .switch {{
                position: relative;
                display: inline-block;
                width: 50px;
                height: 26px;
            }}
            .switch input {{ display: none; }}
            .slider {{
                position: absolute;
                cursor: pointer;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: #bbb;
                transition: 0.2s;
                border-radius: 26px;
            }}
            .slider:before {{
                position: absolute;
                content: "";
                height: 20px; width: 20px;
                left: 3px; bottom: 3px;
                background-color: white;
                transition: 0.2s;
                border-radius: 50%;
            }}
            input:checked + .slider {{
                background-color: #4CAF50;
            }}
            input:checked + .slider:before {{
                transform: translateX(24px);
            }}

            /* === Glowing Color Legend === */
            #colorLegend {{
                position: fixed;
                top: 80px;
                left: 20px;
                background: rgba(255,255,255,0.97);
                padding: 16px 22px;
                border-radius: 16px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                font-size: 15px;
                z-index: 999;
                animation: fadeSlide 1s ease-out forwards, glowPulse 3s ease-in-out infinite;
                border: 1.5px solid rgba(0, 123, 255, 0.2);
            }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 8px;
                font-size: 16px;
                color: #0d47a1;
                text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 6px 0;
                transition: all 0.3s ease;
            }}
            .legend-item.active {{
                transform: scale(1.08);
                font-weight: bold;
                text-shadow: 0 0 5px rgba(13, 71, 161, 0.5);
            }}
            .color-dot {{
                width: 18px;
                height: 18px;
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(0,0,0,0.25);
                border: 1px solid rgba(0,0,0,0.2);
            }}
            @keyframes fadeSlide {{
                from {{ opacity: 0; transform: translateY(-10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes glowPulse {{
                0%,100% {{ box-shadow: 0 0 10px rgba(0,123,255,0.2); }}
                50% {{ box-shadow: 0 0 25px rgba(0,123,255,0.4); }}
            }}
        </style>

        <div id="controlBar">
            <button class="btn" onclick="window.location.href='/'">🏠 Home</button>
            <div class="toggle-group">
                <span>🧠 Brain</span>
                <label class="switch">
                    <input type="checkbox" id="brainToggle" checked>
                    <span class="slider"></span>
                </label>
            </div>
            {"".join([f'<div class="toggle-group"><span>{d}</span><label class="switch"><input type="checkbox" class="tumorToggle" data-idx="{i+1}" checked><span class="slider"></span></label></div>' for i, d in enumerate(day_dirs)])}
            <button class="btn" id="playBtn">▶️ Play</button>
            <button class="btn" id="downloadBtn">📥 Download PNG</button>
        </div>

        <div id="colorLegend">
            <div class="legend-title">🎨 Tumor Days & Colors</div>
            {legend_items}
        </div>

        <script>
            document.addEventListener("DOMContentLoaded", () => {{
                const plot = document.querySelector('.js-plotly-plot');
                const brainToggle = document.getElementById('brainToggle');
                const tumorToggles = document.querySelectorAll('.tumorToggle');
                const playBtn = document.getElementById('playBtn');
                const downloadBtn = document.getElementById('downloadBtn');
                const legendItems = [...document.querySelectorAll('.legend-item')];

                function setVisibility(idx, visible) {{
                    Plotly.restyle(plot, {{ visible: visible ? true : "legendonly" }}, [idx]);
                }}

                brainToggle.addEventListener("change", e => setVisibility(0, e.target.checked));
                tumorToggles.forEach(el => {{
                    el.addEventListener("change", e => {{
                        const traceIdx = parseInt(e.target.dataset.idx);
                        setVisibility(traceIdx, e.target.checked);
                    }});
                }});

                downloadBtn.addEventListener("click", () => {{
                    Plotly.downloadImage(plot, {{
                        format: 'png',
                        filename: 'combined_visualization',
                        height: 800,
                        width: 1000,
                        scale: 2
                    }});
                }});

                // --- Play / Pause Animation with Legend Highlight ---
                let isPlaying = false;
                let currentIdx = 0;
                let interval = null;

                playBtn.addEventListener("click", () => {{
                    if (!isPlaying) {{
                        isPlaying = true;
                        playBtn.textContent = "⏸ Pause";
                        const total = tumorToggles.length;
                        tumorToggles.forEach((el, i) => setVisibility(i+1, false));
                        legendItems.forEach(li => li.classList.remove('active'));
                        interval = setInterval(() => {{
                            if (currentIdx >= total) {{
                                currentIdx = 0;
                                tumorToggles.forEach((el, i) => setVisibility(i+1, false));
                                legendItems.forEach(li => li.classList.remove('active'));
                            }}
                            setVisibility(currentIdx + 1, true);
                            legendItems[currentIdx].classList.add('active');
                            if (currentIdx > 0) legendItems[currentIdx - 1].classList.remove('active');
                            currentIdx++;
                        }}, 1000);
                    }} else {{
                        isPlaying = false;
                        playBtn.textContent = "▶️ Play";
                        clearInterval(interval);
                    }}
                }});
            }});
        </script>
        """

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(toggle_html)

        print(f"[INFO] Combined visualization saved: {out_path}")
        return out_path

    except Exception as e:
        raise RuntimeError(f"Combined visualization generation failed: {e}")
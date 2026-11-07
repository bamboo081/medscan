"""Optional local-only non-clinical demonstration. Install: pip install gradio>=5,<7"""
import argparse
from functools import lru_cache
from pathlib import Path

from predict import load_predictor, classify


def launch():
    p = argparse.ArgumentParser()
    p.add_argument('--mri-checkpoint')
    p.add_argument('--xray-checkpoint')
    p.add_argument('--ct-checkpoint')
    p.add_argument('--port', type=int, default=7860)
    args = p.parse_args()
    paths = {k: v for k, v in [('MRI', args.mri_checkpoint), ('X-ray', args.xray_checkpoint), ('CT', args.ct_checkpoint)] if v}
    if not paths:
        raise SystemExit('Provide at least one --mri-checkpoint, --xray-checkpoint or --ct-checkpoint')
    for f in paths.values():
        if not Path(f).is_file():
            raise FileNotFoundError(f)
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit('Install optional UI: python -m pip install "gradio>=5,<7"') from e

    @lru_cache(maxsize=3)
    def model_for(name):
        selection = {'MRI': 'mri', 'X-ray': 'xray', 'CT': 'ct'}[name]
        return load_predictor(paths[name], modality=selection)

    def run(file_path, modality):
        if file_path is None:
            raise gr.Error('Choose a local sample image.')
        try:
            return classify(*model_for(modality), file_path)
        except (OSError, RuntimeError, ValueError) as e:
            raise gr.Error(str(e)) from e

    with gr.Blocks(title='MedScan — Non-clinical research demo', analytics_enabled=False) as demo:
        gr.Markdown('# MedScan · Research Demonstration\nExperimental medical-image classification. **Not for clinical decisions.** Do not upload identifiable patient data.')
        modality = gr.Dropdown(choices=list(paths), value=next(iter(paths)), label='Choose scan modality (not detected automatically)')
        file_path = gr.File(label='Research sample image (.png/.jpg/.npy)', file_types=['.png', '.jpg', '.jpeg', '.npy'], type='filepath')
        submit = gr.Button('Analyze sample')
        output = gr.JSON(label='Uncalibrated research scores and limitations')
        submit.click(run, inputs=[file_path, modality], outputs=output)
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False)


if __name__ == '__main__':
    launch()

import os
import sys
import glob
import importlib.util

# Force non-interactive backend BEFORE importing pyplot to stop freeze/popups
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Mock IPython display to ignore HTML(ani.to_jshtml()) calls in .py scripts
import builtins
try:
    import IPython.display
    IPython.display.HTML = lambda *args, **kwargs: None
    IPython.display.clear_output = lambda *args, **kwargs: None
except ImportError:
    pass

def list_python_files():
    current_script = os.path.basename(__file__)
    return sorted([f for f in glob.glob("*.py") if f != current_script])

def import_and_run(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Suppress plt.show during module execution
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    
    plt.close('all')
    print(f"\n[+] Executing '{file_path}'...")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    finally:
        plt.show = original_show
        
    return module

def save_and_open_video(anim, filename):
    print(f"\n[+] Rendering MP4 video to '{filename}' using FFmpeg...")
    
    # Disable frame caching to fix memory leak and 'WARN: COPY MODE'
    anim._cache_frame_data = False
    
    writer = animation.FFMpegWriter(
        fps=20, 
        metadata=dict(artist='Phase Field Runner'), 
        bitrate=1800
    )
    
    try:
        anim.save(filename, writer=writer)
        print(f"[✓] Successfully saved video: {filename}")
        
        if sys.platform.startswith('darwin'):
            os.system(f'open "{filename}"')
        elif os.name == 'nt':
            os.startfile(filename)
        elif os.name == 'posix':
            os.system(f'xdg-open "{filename}" &')
            
    except Exception as e:
        print(f"[!] Video encoding failed: {e}")

def main():
    while True:
        py_files = list_python_files()
        if not py_files:
            print("No Python simulation scripts found.")
            return

        print("\n==============================================")
        print("    PHASE FIELD MODEL INTERACTIVE RUNNER      ")
        print("==============================================")
        for idx, fname in enumerate(py_files, 1):
            print(f" [{idx}] {fname}")
        print(" [0] Exit")
        print("----------------------------------------------")

        choice = input("Select a simulation script to run: ").strip()
        if choice == '0':
            break

        if not choice.isdigit() or not (1 <= int(choice) <= len(py_files)):
            print("Invalid selection. Try again.")
            continue

        selected_file = py_files[int(choice) - 1]
        mod = import_and_run(selected_file)

        # Search for FuncAnimation instance created by target file
        found_anim = None
        for attr in dir(mod):
            val = getattr(mod, attr)
            if isinstance(val, animation.FuncAnimation):
                found_anim = val
                break

        print("\n----------------------------------------------")
        print("Render Options:")
        print(" [1] Show Final Plot")
        print(" [2] Render Animation as .mp4 Video")
        print(" [3] Both Plot & .mp4 Video")
        
        render_mode = input("Select output (1-3): ").strip()

        if render_mode in ['2', '3']:
            if found_anim:
                output_video = f"{os.path.splitext(selected_file)[0]}_output.mp4"
                save_and_open_video(found_anim, output_video)
            else:
                print("\n[!] No `FuncAnimation` object was detected in this script.")

        if render_mode in ['1', '3']:
            try:
                matplotlib.use('TkAgg')
                fig = plt.gcf()
                if fig and fig.get_axes():
                    plt.show()
                else:
                    print("\n[!] No active matplotlib figure to display.")
            except Exception as e:
                print(f"[!] Could not display GUI plot: {e}")

        input("\nPress [Enter] to return to menu...")

if __name__ == "__main__":
    main()
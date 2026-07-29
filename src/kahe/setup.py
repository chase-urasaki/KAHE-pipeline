"""Setup script to initialize a KAHE pipeline project."""
import shutil
import argparse
from pathlib import Path


PROJECT_SUBDIRS = [
    "cals",
    "raw",
    "extracted_spectra",
    "wl_calibrated_spectra",
    "telluric_correction_inputs",
    "telluric_correction_out",
]


def setup_project(project_name: str, output_base: str = ".") -> None:
    """
    Initialize a new KAHE project with directory structure and blank config.
    
    Args:
        project_name: Name of project (e.g., 'NGC1234_2024-07-15')
        output_base: Base directory to create project in (default: current dir)
    """
    # Create project directory
    project_dir = Path(output_base) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all subdirectories
    for subdir in PROJECT_SUBDIRS:
        (project_dir / subdir).mkdir(exist_ok=True)
        print(f"Created {subdir}/")
    
    # Copy blank config template to project root
    template_path = Path(__file__).parent / "template_config.ini"
    config_output = project_dir / "pipeline_config.ini"
    
    if template_path.exists():
        shutil.copy(template_path, config_output)
        print(f"Created pipeline_config.ini")
    else:
        print(f"Warning: template not found at {template_path}")
    
    print(f"Project '{project_name}' initialized at {project_dir}")
    print(f"Edit config: {config_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new KAHE pipeline project"
    )
    parser.add_argument(
        "project_name",
        help="Project name (e.g., 'TOI2123_240704')"
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output base directory (default: current directory)"
    )
    
    args = parser.parse_args()
    setup_project(args.project_name, args.output)


if __name__ == "__main__":
    main()
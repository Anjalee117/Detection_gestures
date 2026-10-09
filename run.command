#!/bin/zsh
project_dir="${0:A:h}"
cd "$project_dir" || exit 1
if [[ -x "$project_dir/.venv/bin/python" ]]; then
    app_python="$project_dir/.venv/bin/python"
elif [[ -x "$project_dir/venv/bin/python" ]]; then
    app_python="$project_dir/venv/bin/python"
else
    print 'Create a virtual environment and install requirements.txt; see SETUP.md.'
    read -k 1
    exit 1
fi
exec "$app_python" "$project_dir/app.py"

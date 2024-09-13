#! /bin/bash

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "./scripts/driver.bashrc"



RUN_TAG="all"
BUILD_TAG="all"
build=false
run=false

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir=${CURR_DIR}/log/$current_time
if [[ ! -d $log_dir ]]; then
    mkdir -p $log_dir
fi


function show_usage() {
    cat <<EOF
Usage: $0 [options] ...
OPTIONS:
    -h, --help             Display this help and exit.
    -b, --build            Build proj env, e.g.: ./build.sh -b
    -r, --run              Run proj app with background operation, e.g.: ./build.sh -r

EOF
}


function parse_arguments() {
    local custom_version=""
    local custom_dist=""
    local shm_size=""
    local geo=""

    while [ $# -gt 0 ]; do
        local opt="$1"
        shift
        case "${opt}" in
            -h | --help)
                show_usage
                exit 1
                ;;
            -b | --build)
                build=true
                ;;
            -r | --run)
                run=true
                ;;
            *)
                warning "Unknown option: ${opt}"
                exit 2
                ;;
        esac
    done # End while loop
}


function check_requirements_for_proj() {
    info "==========install requirements.txt for proj=========="
    local packages="$1"
    wheel_list=(pyyaml openpyxl flask matplotlib pandas==1.4.2 scikit-learn statsmodels pmdarima)
    for item in "${wheel_list[@]}"
    do
      if echo "${packages[@]}" | grep -qw "$item"; then
        warning_green "$item is installed"
      else
        info "install $item..."
        # sudo python3.9 -m pip install --no-index --find-links=$packages_dir/wheel/pytorch/ $item > $log_dir/install_$item.log 2>&1
        # python3.9 -m pip install --no-index --find-links=$packages_dir/wheel/pytorch/ $item 2>&1 | tee $log_dir/pip_install_$item.log
        python3.8 -m pip install $item -i https://pypi.tuna.tsinghua.edu.cn/simple
      fi
    done
}




function main() {

    if [ -f /.dockerenv ]; then
        info "in Container environment"
    else
        info "in Physical machine environment"
    fi

    parse_arguments "$@"

    set -e

    if $build; then
        pip_packages=$(pip list --format=freeze | cut -d'=' -f1)

        if [[ "${BUILD_TAG}" == "all" ]]; then
            check_requirements_for_proj "$pip_packages"
            ok "build pass ..."
        else
            show_usage
        fi
    fi

    if $run; then
        nohup python3.8 -u app.py > $log_dir/nohup.out 2>&1 &

        sleep 3

        file_count=$(ls -A "$log_dir" | wc -l)
        if [ "$file_count" -eq 0 ]; then
            rm -rf $log_dir
            ok "run failed, please debug ..."
        else
            ok "Check the log in the folder $log_dir, run ' tail -f $log_dir/nohup.out '"
        fi
    fi
}


main "$@"
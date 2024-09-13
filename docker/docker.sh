#!/bin/bash
path=$(cd "$(dirname "$0")";pwd)
export WORKSPACE_PATH="$path/.."
export DOCKER_USERS=$USER
export USER=$DOCKER_USERS
export USER_ID="$(id -u $DOCKER_USERS)"
export GROUP_ID="$(id -g $DOCKER_USERS)"
CI_USER="qa"
hardware=`uname -m`
device_name="prediction"
release_compose_file="docker-compose.yaml"
dev_compose_file="docker-compose-dev.yaml"
compose_file=${release_compose_file}
if [ "${hardware}" == "x86_64" ]; then
    device_name="prediction-dev"
    compose_file=${dev_compose_file}
fi

container_name="cmat-imbr-algorithm"

function start_qa()
{
    container=`docker ps | grep ${device_name}-${CI_USER} | awk '{print $10}'`
    if [ "$container" != "${device_name}-${CI_USER}" ]; then
        export DOCKER_USERS=$CI_USER
        export USER_ID=1010
        export GROUP_ID=1010
	sed -i "s/ros2:/ros2-${CI_USER}: /" $path/${compose_file}
        docker-compose -f $path/${compose_file} up -d
	docker cp $path/env.sh ${device_name}-${CI_USER}:/tmp
	docker exec -u root ${device_name}-${CI_USER} \
            bash -c '/tmp/env.sh'
        exit 0
    fi

    echo "${device_name}-${CI_USER} already start."
}

function init_docker()
{
    docker cp $path/env.sh $1:/tmp
    if [ "${USER}" != "root" ]; then
        docker exec -u root "$1" \
            bash -c '/tmp/env.sh'
    fi
}

function into_docker()
{
    xhost +local:root 1>/dev/null 2>&1
    docker exec -it -u ${USER} ${container_name} bash
}

function start_docker()
{
    container=`docker ps | grep ${container_name} | awk '{print $10}'`
    if [ "$container" != "${container_name}" ]; then
	sed -i "s/ros2:/ros2-${USER}: /" $path/${compose_file}
        docker-compose -f $path/${compose_file} up -d ros2-${USER}  # 启动指定services，避免将所有的docker-compose container启动
	init_docker ${container_name}
	into_docker
	exit 0
    fi

    echo "${container_name} already start."
}

function restart_docker()
{
    docker-compose -f $path/${compose_file} up -d
    init_docker ${container_name}
}

function stop_qa()
{
    docker-compose -f $path/${compose_file} down -v
    sed -i "s/ros2-${CI_USER}: /ros2:/" $path/${compose_file}
}


function stop_docker()
{
    docker-compose -f $path/${compose_file} down -v
}


function help()
{
    cat << EOF
Usage: COMMAND [<function>]

COMMANDS:
  help:      show this help message
  start:     start the docker
  stop:      stop the docker
  restart:   restart the docker
  into:      into the docker with $USER
EOF
}

function main()
{
    local cmd=$1
    case $cmd in
        start)
	    start_docker
	    ;;
	start_qa)
	    start_qa
	    ;;
	stop)
	    stop_docker
	    ;;
    stop_qa)
        stop_qa
        ;;
	restart)
	    restart_docker
	    ;;
	into)
	    into_docker
	    ;;
	-h|--help|help)
	    help
	    ;;
	*)
	    help
	    ;;
    esac
}

main "$@"

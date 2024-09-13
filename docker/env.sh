#!/bin/bash
MACHINE=`uname -m`

function create_user_account() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"
  addgroup --gid "${gid}" "${group_name}"
  useradd ${user_name} -u ${uid} -g ${gid} -m
  echo "%sudo  ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/nopasswd
  usermod -aG sudo "${user_name}"
  usermod -aG video "${user_name}"
  chown -R "${user_name}":"${user_name}" /workspace
}

function setup_user_account_if_not_exist() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"
  if grep -q "^${user_name}:" /etc/passwd; then
      echo "User ${user_name} already exist. Skip setting user account."
      return
  fi
  create_user_account "$@"
  # echo "127.0.0.1 in-dev-docker" >> /etc/hosts
}

function grant_device_permissions() {
  echo "todo: grant_device_permissions"
}

function setup_core_pattern() {
  if [[ -w /proc/sys/kernel/core_pattern ]]; then
      echo "/workspace/data/core/core_%e.%p" > /proc/sys/kernel/core_pattern
  fi
}

function main() {
  local user_name="$1"
  local uid="$2"
  local group_name="$3"
  local gid="$4"

  if [ "${uid}" != "${gid}" ]; then
    echo "Warning: uid(${uid}) != gid(${gid}) found."
  fi
  if [ "${user_name}" != "${group_name}" ]; then
    echo "Warning: user_name(${user_name}) != group_name(${group_name}) found."
  fi

  setup_user_account_if_not_exist "$@"
  setup_core_pattern
  echo "PS1='\[\033[01;32m\]\u@\[\033[01;35m\]\h\[\033[00m\]:\[\033[01;36m\]\w\[\033[00m\]$ '" >> /home/$1/.bashrc
  # vim utf-8
  echo "set encoding=utf-8" >> /home/$1/.vimrc
  echo "set fileencodings=utf-8" >> /home/$1/.vimrc

  cd /workspace
  ./build.sh -r
}

main "${DOCKER_USER}" "${DOCKER_USER_ID}" "${DOCKER_GRP}" "${DOCKER_GRP_ID}"

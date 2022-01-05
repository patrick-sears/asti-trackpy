#!/bin/bash


wd=`pwd -P`


cd "$HOME/bin"

lname="prs-asti-trackpy"

if [[ -L "$lname" ]]; then
  unlink "$lname"
fi

ln -sf "$wd/main.py" "$lname"




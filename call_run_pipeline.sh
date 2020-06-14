#!/bin/bash

# used to simultaneously run different experiments from the same script
# https://stackoverflow.com/questions/3398258/edit-shell-script-while-its-running/3399850#3399850

called() {
	source run_pipeline.sh
}
called

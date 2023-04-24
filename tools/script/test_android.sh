#!/bin/bash
adb push ./input_0.txt /data/local/tmp/ace/input_0.txt
adb push ./input_1.txt /data/local/tmp/ace/input_1.txt
adb push ./temp.bin /data/local/tmp/ace/temp.bin

./MNNV2Basic.out temp.bin

adb pull /data/local/tmp/ace/output.txt output_android.txt

#coding=utf8

# Copyright (c) 2017, Baidu.com, Inc. All Rights Reserved
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import sys
from familia_wrapper import InferenceEngineWrapper

if sys.version_info < (3,0):
    input = raw_input

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage:python {sys.argv[0]} model_dir conf_file\n")
        exit(-1)
    # 获取参数
    model_dir = sys.argv[1]
    conf_file = sys.argv[2]
    # 创建InferenceEngineWrapper对象
    inference_engine_wrapper = InferenceEngineWrapper(model_dir, conf_file)
    while True:
        input_text = input("Enter Document: ")
        # 分词
        seg_list = inference_engine_wrapper.tokenize(input_text.strip())
        length = len(seg_list)
        sentences = [seg_list[index: index + 5] for index in range(0, length, 5)]
        # 进行推断
        topic_dist = inference_engine_wrapper.slda_infer(sentences)
        # 打印结果
        print("Document Topic Distribution:")
        print(topic_dist)

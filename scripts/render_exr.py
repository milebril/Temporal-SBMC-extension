#!/usr/bin/env python
# encoding: utf-8
# Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
# Michaël Gharbi Tzu-Mao Li Miika Aittala Jaakko Lehtinen Frédo Durand
# Siggraph 2019
#
# Copyright (c) 2019 Michaël Gharbi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Renders a .pbrt scene to a .exr image."""
from sbmc import rendering

if __name__ == "__main__":
    parser = rendering.PBRTRenderingParser(
        description="Render a .pbrt scene file using the default pathtracer"
        " in PBRTv2.")
    rendering.PBRTRenderer(parser.parse_args()).render()

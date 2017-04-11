# Copyright (C) 2016 Nigel Williams
#
# Vulkan Free Surface Modeling Engine (VFSME) is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

FLAGS=-std=gnu++14 -Weverything -Wno-c++98-compat
OPT=-Ofast -mfma -mavx2

.PHONY: clean

bitmap_rotate: bitmap_rotate.cpp
	clang++ $(FLAGS) $(OPT) bitmap_rotate.cpp -o $@

profile_guided_optimization: bitmap_rotate.cpp
	clang++ -gline-tables-only -std=gnu++14 -Ofast bitmap_rotate.cpp -o bitmap_rotate
	sudo perf record -b ./bitmap_rotate earth.pgm 45
	sudo ../tools/autofdo/create_llvm_prof --binary=./bitmap_rotate --out=bitmap_rotate.prof
	clang++ -gline-tables-only -std=gnu++14 -Ofast -fprofile-sample-use=bitmap_rotate.prof bitmap_rotate.cpp -o bitmap_rotate

clean:
	rm bitmap_rotate

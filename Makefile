# IGIE_ROOT : tvm package path
IGIE_ROOT=$(shell pip3 show igie | grep Location | cut -d " " -f2)/tvm/

PKG_CFLAGS= -std=c++17 -O3 -fPIC\
        -I${IGIE_ROOT}/include\
        -I/usr/include/\
        -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>\
        -D_GLIBCXX_USE_CXX11_ABI=0\

PKG_LDFLAGS = -L${IGIE_ROOT} -ldl -pthread

.PHONY: clean all

#all: deploy
all: deploy deploy_profiling_time

# Deploy using pre-built libtvm_runtime.so
deploy: deploy.cc 
	@echo "IGIE_ROOT=$(IGIE_ROOT)"
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

deploy_profiling_time: deploy_profiling_time.cc 
	@echo "IGIE_ROOT=$(IGIE_ROOT)"
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)

clean:
	rm -rf deploy

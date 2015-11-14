include ${HOME}/libs/conf/ElVars

main: obj/chase.o obj/main.o
	${CXX} ${EL_COMPILE_FLAGS} obj/main.o obj/chase.o -o $@ ${EL_LINK_FLAGS} ${EL_LIBS}

libchase.a: obj/chase.o obj/elechfsi_fleur.o
	ar -rcs $@ obj/chase.o obj/elechfsi_fleur.o

obj/chase.o: src/chase.cpp
	${CXX} ${EL_COMPILE_FLAGS} -c $< -o $@

obj/main.o: src/main.cpp
	${CXX} ${EL_COMPILE_FLAGS} -c $< -o $@

obj/elechfsi_fleur.o: src/elechfsi_fleur.cpp
	${CXX} ${EL_COMPILE_FLAGS} -c $< -o $@

filter: src/driver_filter.cpp
	${CXX} ${EL_COMPILE_FLAGS} $< -o $@ ${EL_LINK_FLAGS} ${EL_LIBS}

clean:
	rm -f obj/* main chase.a

obj/chase_f.o: chase.F90
	gfortran -c chase.F90  -o $@

install:
	cp libchase.a ~/FleurChase/libs/lib

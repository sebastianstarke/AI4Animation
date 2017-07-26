PLATFORM = $(shell uname)

ifeq ($(findstring Linux,$(PLATFORM)),Linux)
	LFLAGS = -lGL -lGLEW -lSDL2
	EXT = 
endif

ifeq ($(findstring Darwin,$(PLATFORM)),Darwin)
	LFLAGS = -lGL -lGLEW -lSDL2
	EXT = 
endif

ifeq ($(findstring MINGW,$(PLATFORM)),MINGW)
	LFLAGS = -lmingw32 -lopengl32 -lglew32 -lSDL2main -lSDL2 
	EXT = .exe
endif

pfnn: pfnn.cpp
	g++ -std=gnu++11 -Wall -O3 -ffast-math $< $(LFLAGS) -g -o $@
	g++ -DHIGH_QUALITY -std=gnu++11 -Wall -O3 -ffast-math $< $(LFLAGS) -g -o $@_hq
  
clean: pfnn$(EXT)
	rm pfnn$(EXT)
	rm pfnn_hq$(EXT)
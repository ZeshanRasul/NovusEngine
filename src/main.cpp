#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include "core/engine_app.h"

int main()
{
	try
	{
       EngineApp app;
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

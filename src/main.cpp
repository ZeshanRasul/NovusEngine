#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include "renderer/renderer.h"

int main()
{
	try
	{
		Renderer app;
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

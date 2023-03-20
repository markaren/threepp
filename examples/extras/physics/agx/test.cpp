#include <agxSDK/Simulation.h>
#include <agx/RigidBody.h>

int main()
{
    try
    {
        // Where AGX is placed/installed:
        // For example "c:\\Program Files\\Algoryx\\AGX-2.33.0.0\\";
        const char* path = getenv("AGX_DIR");
        std::string pathToAGX;

        if (path)
            std::string pathToAGX = path;
        else
            std::cerr << "*** Environment variable AGX_DIR is not set. This might lead to problems during runtime" << std::endl;

        // Probably where the license file is
        agxIO::Environment::instance()->getFilePath(
                                              agxIO::Environment::RESOURCE_PATH).pushbackPath(pathToAGX);

        // Text files for plugins
        agxIO::Environment::instance()->getFilePath(
                                              agxIO::Environment::RESOURCE_PATH).addFilePath(pathToAGX);

        // binary plugin files
        agxIO::Environment::instance()->getFilePath(
                                              agxIO::Environment::RUNTIME_PATH).addFilePath(pathToAGX + "/bin/x64/plugins");

        // resource files
        agxIO::Environment::instance()->getFilePath(
                                              agxIO::Environment::RESOURCE_PATH).addFilePath(pathToAGX + "/data");

        // AutoInit will call agx::init() which must be called before
        // using the AGX API creating resources such as bodies, geometries etc.
        std::cout << "*** Initializing AGX Dynamics..." << std::endl;
        agx::AutoInit init;
        {
            std::cout << "*** Creating a simulation" << std::endl;

            // Create a Simulation which holds the DynamicsSystem and Space.
            agxSDK::SimulationRef sim = new agxSDK::Simulation;

            std::cout << "*** Creating a RigidBody" << std::endl;

            // Create a rigid body (no geometry) with default mass etc.
            agx::RigidBodyRef body = new agx::RigidBody;

            // Add the body to the simulation.
            sim->add(body);

            // Set the time step to 1/100 (0.01 s or 100hz):
            sim->setTimeStep(1.0 / 60);

            // Simulate for some time
            std::cout << "*** Simulating" << std::endl;

            while (sim->getTimeStamp() < 0.5)
            {
                sim->stepForward();     // Take one time step.
                std::cout << "   " << sim->getTimeStamp() << ": \t" << body->getPosition() << std::endl;
            }
        }
        // The destructor for AutoInit will call agx::shutdown() automatically.
        // Unloads plugins, destroys threads etc.
        std::cout << "*** Un-initializing AGX Dynamics..." << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "*** Caught an exception: " << e.what() << std::endl;
    }
    return 0;
}
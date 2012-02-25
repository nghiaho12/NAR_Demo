#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

#include "VideoThread.h"

#ifdef _WIN32
	#include <irrlicht.h>
#elif __linux__
	#include <irrlicht/irrlicht.h>
#else
	#error "Platform not supported or tested"
#endif

#ifdef _IRR_WINDOWS_
#pragma comment(lib, "Irrlicht.lib")
#pragma comment(linker, "/subsystem:console /ENTRY:mainCRTStartup")
#endif

#ifdef _WIN32
	#ifdef _DEBUG
	#pragma comment(lib, "opencv_calib3d231d.lib")
	#pragma comment(lib, "opencv_core231d.lib")
	#pragma comment(lib, "opencv_features2d231d.lib")
	#pragma comment(lib, "opencv_highgui231d.lib")
	#pragma comment(lib, "opencv_imgproc231d.lib")
	#pragma comment(lib, "opencv_ml231d.lib")
	#pragma comment(lib, "opencv_video231d.lib")
#else
	#pragma comment(lib, "opencv_calib3d231.lib")
	#pragma comment(lib, "opencv_core231.lib")
	#pragma comment(lib, "opencv_features2d231.lib")
	#pragma comment(lib, "opencv_highgui231.lib")
	#pragma comment(lib, "opencv_imgproc231.lib")
	#pragma comment(lib, "opencv_ml231.lib")
	#pragma comment(lib, "opencv_video231.lib")
	#endif
#endif

using namespace std;

using namespace irr;
using namespace core;
using namespace scene;
using namespace video;
using namespace io;
using namespace gui;

const int VIDEO_WIDTH = 640;
const int VIDEO_HEIGHT = 480;

bool g_running = true;

// Process keyboard event
class MyEventReceiver : public IEventReceiver
{
public:
    // This is the one method that we have to implement
    virtual bool OnEvent(const SEvent& event)
    {
        if (event.EventType == irr::EET_KEY_INPUT_EVENT) {
            if(event.KeyInput.Key == KEY_ESCAPE) {
                g_running = false;
            }
        }

        return false;
    }

};

int main()
{
    VideoThread video_thread;

    // Initialisation
    MyEventReceiver reciever;

    IrrlichtDevice *device = createDevice(video::EDT_OPENGL, dimension2d<u32>(VIDEO_WIDTH, VIDEO_HEIGHT), 16, false, true /* shadow */, false, &reciever);

    if (!device) {
        cerr << "OpenGL support not found :(\n" << endl;
        return -1;
    }

    device->setWindowCaption(L"NAR Demo!");

    IVideoDriver *driver = device->getVideoDriver();
    ISceneManager *smgr = device->getSceneManager();
    IGUIEnvironment *guienv = device->getGUIEnvironment();

    IAnimatedMesh *mesh = smgr->getMesh("media/sydney.md2");

    // Texture where we will store video frames
    // NOTE: ECF_R8G8B8 doesn't work
    // RGBA format
    ITexture *tex = driver->addTexture(vector2d<u32>(VIDEO_WIDTH, VIDEO_HEIGHT), "video_stream");

    if(!mesh) {
		cerr << "Can't find mesh" << endl;
        device->drop();
        return -1;
    }

    // Setup the threads
    {
        cv::Mat ARObject = cv::imread("media/AR_object.png");

        if(!ARObject.data) {
            cerr << "media/AR_object.png not found" << endl;
            return -1;
        }

        // Give the process thread access to the video thread
        video_thread.GetNAR().SetARObject(ARObject);
        video_thread.Run();
    }

    // Font for drawing text
    gui::IGUIFont* font = device->getGUIEnvironment()->getFont("media/bitstream_font.xml");
    assert(font);

    // Lighting
    ILightSceneNode* light1 = smgr->addLightSceneNode(0, core::vector3df(-10, 10, -10), video::SColorf(1.0f,1.0f,1.0f));
    smgr->setAmbientLight(video::SColorf(1.0f,1.0f,1.0f));

    // Setup camera - remains static at the origin, +z is pointing into the screen! differnt to OpenGL
    ICameraSceneNode *camera = smgr->addCameraSceneNode(0, vector3df(0,0,0), vector3df(0,0,1));
    camera->setFOV((float)(60.0 * M_PI/180.0));

    // Add 3-axis
    scene::ISceneNode *y_axis = smgr->addMeshSceneNode(
            smgr->addArrowMesh("y-axis",
                            video::SColor(255, 0, 200, 0),
                            video::SColor(255, 0, 255, 0),
                            4, 8, 0.5f, 0.4f, 0.05f, 0.1f)
            );

    {

        scene::ISceneNode *x_axis = smgr->addMeshSceneNode(
                smgr->addArrowMesh("x-axis",
                                video::SColor(255, 200, 0, 0),
                                video::SColor(255, 255, 0, 0),
                                4, 8, 0.5f, 0.4f, 0.05f, 0.1f)
                );

        scene::ISceneNode *z_axis = smgr->addMeshSceneNode(
                smgr->addArrowMesh("z-axis",
                                video::SColor(255, 0, 0, 200),
                                video::SColor(255, 0, 0, 255),
                                4, 8, 0.5f, 0.4f, 0.05f, 0.1f)
                );

        y_axis->addChild(x_axis);
        y_axis->addChild(z_axis);

        // x and z are children of y
        x_axis->setMaterialFlag(video::EMF_LIGHTING, false);
        x_axis->setPosition(core::vector3df(0, 0, 0));
        x_axis->setRotation(core::vector3df(0, 0, -90));

        z_axis->setMaterialFlag(video::EMF_LIGHTING, false);
        z_axis->setPosition(core::vector3df(0, 0, 0));
        z_axis->setRotation(core::vector3df(90, 0, 0));

        y_axis->setMaterialFlag(video::EMF_LIGHTING, false);
        y_axis->setVisible(false);
    }

    // Add the model to the scene
    // I use an empty scene to allow for prior transformation applied to the model.
    // Need to rotate and translate the model so the feet at the ground are at 0.
    scene::ISceneNode *person = smgr->addEmptySceneNode();
    {
        IAnimatedMeshSceneNode *model = smgr->addAnimatedMeshSceneNode(mesh);

        person->addChild(model);

        person->setMaterialFlag(video::EMF_LIGHTING, true);

        model->setMaterialFlag(EMF_LIGHTING, true);
        model->setMD2Animation(scene::EMAT_STAND);
        model->setMaterialTexture(0, driver->getTexture("media/sydney.bmp"));
        model->addShadowVolumeSceneNode();

        model->setRotation(core::vector3df(0, 0, 0));

        float scale = 0.020f;

        // Find the offset to the ground, where the model stands on
        core::aabbox3d<f32> box = model->getBoundingBox();

        float person_ground_offset = box.MinEdge.Y*scale;

        model->setPosition(core::vector3df(0, -person_ground_offset, 0));
        model->setScale(core::vector3df(scale, scale, scale));
    }

    // Attach a flying orb around the person
    {
        // create light
        ILightSceneNode *light = smgr->addLightSceneNode(0, core::vector3df(0,0,0.0), video::SColorf(1.0f, 0.6f, 0.7f, 1.0f), 800.0f);
        scene::ISceneNodeAnimator *anim = smgr->createFlyCircleAnimator (core::vector3df(0,1.0,0), 0.5f, 0.002f);
        light->addAnimator(anim);
        anim->drop();

        // attach billboard to light
        IBillboardSceneNode *billboard = smgr->addBillboardSceneNode(light, core::dimension2d<f32>(0.4f, 0.4f));
        billboard->setMaterialFlag(video::EMF_LIGHTING, false);
        billboard->setMaterialType(video::EMT_TRANSPARENT_ADD_COLOR);
        billboard->setMaterialTexture(0, driver->getTexture("media/particlewhite.bmp"));

        person->addChild(light);
    }

    ThreadJob last_job;
    bool job_init = false;

    while(device->run() && g_running) {
        driver->beginScene();

        // AR part
        boost::mutex::scoped_lock lock(video_thread.GetNAR().m_job_mutex);

        if(!video_thread.GetNAR().GetJobsDone().empty()) {
            last_job = video_thread.GetNAR().GetJobsDone().front();
            video_thread.GetNAR().GetJobsDone().pop_front();
            lock.unlock();

            job_init = true;

            // Image
            unsigned char *tex_buf = (unsigned char*)tex->lock();
            unsigned char *frame_buf = last_job.img.data;

            // Convert from RGB to RGBA
            for(int y=0; y < last_job.img.rows; y++) {
                for(int x=0; x < last_job.img.cols; x++) {
                    *(tex_buf++) = *(frame_buf++);
                    *(tex_buf++) = *(frame_buf++);
                    *(tex_buf++) = *(frame_buf++);
                    *(tex_buf++) = 255;
                }
            }

            tex->unlock();
        }
        else {
            lock.unlock();
        }

        driver->draw2DImage(tex, core::rect<s32>(0,0,VIDEO_WIDTH,VIDEO_HEIGHT), core::rect<s32>(0,0,VIDEO_WIDTH,VIDEO_HEIGHT));

        if(job_init) {
            if(last_job.status == NAR::GOOD) {
                cv::Mat t = last_job.translation;
                cv::Mat R = last_job.rotation;

                float tx = (float)t.at<double>(0,0);
                float ty = (float)-t.at<double>(1,0);
                float tz = (float)t.at<double>(2,0);

                double yaw, pitch, roll;
                video_thread.GetNAR().GetYPR(R, yaw, pitch, roll);

                // bit of guess work ...
                yaw = -TO_DEG(yaw);
                pitch = TO_DEG(pitch);
                roll = -TO_DEG(roll);

                person->setPosition(core::vector3df(tx, ty, tz));
                person->setRotation(core::vector3df((float)roll - 90.0f, (float)pitch, (float)yaw));

                y_axis->setPosition(core::vector3df(tx, ty, tz));
                y_axis->setRotation(core::vector3df((float)roll, (float)pitch, (float)yaw));

                person->setVisible(true);
                y_axis->setVisible(true);
            }
            else if(last_job.status == NAR::BAD) {
                person->setVisible(false);
                y_axis->setVisible(false);
            }

            if(last_job.status != NAR::BAD) {
                // Draw the features matched
                for(size_t i=0; i < last_job.matches.size(); i++) {
                    int x = (int)(last_job.matches[i].x + 0.5f);
                    int y = (int)(last_job.matches[i].y + 0.5f);
                    //float scale = last_job.matches[i].scale;
                   // float radius = (NAR_PATCH_SIZE/2) / scale;

                    driver->draw2DPolygon(core::position2d<s32>(x,y), 8, SColor(100, 255, 0, 0), 32);
                }

                // Draw the countour
                for(int i=0; i < 4; i++) {
                    cv::Point2i &pt1 = last_job.corners[i];
                    cv::Point2i &pt2 = last_job.corners[(i+1)%4];
                    driver->draw2DLine(core::position2d<s32>(pt1.x, pt1.y), core::position2d<s32>(pt2.x, pt2.y), SColor(255,255,0,0));
                }
                /*
                // Draw the search region
                if(last_job.use_search_region) {
                    cv::Point2i &pt1 =  last_job.search_region_start;
                    cv::Point2i &pt2 =  last_job.search_region_end;

                    driver->draw2DRectangleOutline(core::recti(pt1.x, pt1.y, pt2.x, pt2.y), SColor(255,0,255,0));
                }
                */

                for(size_t i=0; i < last_job.optical_flow_tracks.size(); i++) {
                    cv::Point2f &pt = last_job.optical_flow_tracks[i];
					int x = (int)(pt.x + 0.5f);
					int y = (int)(pt.y + 0.5f);
                    driver->draw2DPolygon(core::position2d<s32>(x,y), 3, SColor(100, 0, 255, 0), 32);
                }
            }
        }

        if (font) {
            driver->draw2DRectangle(video::SColor(200, 0, 0, 0), core::rect<s32>(0,0, 250, 85));

            stringstream text;
            int ypos = 010;
            int yincr = 15;

            text << fixed << setprecision(2) << "AR signature (bytes): " << video_thread.GetNAR().GetARObjectSigSizeBytes();
            font->draw(text.str().c_str(), core::rect<s32>(10,ypos,300,50), video::SColor(255,0,255,0));
            ypos += yincr;

            text.str("");
            text << fixed << setprecision(2) << "Video FPS: " << video_thread.GetFPS();
            font->draw(text.str().c_str(), core::rect<s32>(10,ypos,300,50), video::SColor(255,0,255,0));
            ypos += yincr;

            text.str("");
            text << fixed << setprecision(2) << "Processing FPS: " << video_thread.GetNAR().GetFPS();
            font->draw(text.str().c_str(), core::rect<s32>(10,ypos,300,50), video::SColor(255,0,255,0));
            ypos += yincr;

            text.str("");
            text << fixed << setprecision(2) << "Status: ";

            if(last_job.status == NAR::GOOD) {
                text << "Detected";
            }
            else if(last_job.status == NAR::PREDICTION) {
                text << "Prediction";
            }
            else {
                text << "Searching";
            }

            font->draw(text.str().c_str(), core::rect<s32>(10,ypos,300,50), video::SColor(255,0,255,0));
        }

        smgr->drawAll();
        guienv->drawAll();

        driver->endScene();
    }

    device->drop();

    video_thread.Done();

    return 0;
}

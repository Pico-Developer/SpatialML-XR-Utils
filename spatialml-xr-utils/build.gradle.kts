plugins {
    id("com.android.library")
}

android {
    namespace = "com.pico.spatialml.xrutils"
    compileSdk = 34
    ndkVersion = "26.3.11579264"

    defaultConfig {
        minSdk = 34
        consumerProguardFiles("consumer-rules.pro")
    }

    lint {
        disable.add("ExpiredTargetSdkVersion")
    }
}

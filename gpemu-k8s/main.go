package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/fsnotify/fsnotify"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	ConfigFilePath              = "/gpemu-k8s-device-plugin/config.json"
	DefaultLocalConfigFilePath  = "config.json"
	DevicePluginPathEnvVar      = "GPEMU_DEVICE_PLUGIN_PATH"
	ConfigFilePathEnvVar        = "GPEMU_CONFIG_FILE"
	DefaultFallbackResourceName = "gpemu.dev/egpu"
)

func resolveDevicePluginPath() string {
	if p := strings.TrimSpace(os.Getenv(DevicePluginPathEnvVar)); p != "" {
		return p
	}

	if st, err := os.Stat(pluginapi.DevicePluginPath); err == nil && st.IsDir() {
		return pluginapi.DevicePluginPath
	}

	return filepath.Join(os.TempDir(), "gpemu-device-plugins")
}

func ensureDevicePluginPath() (string, error) {
	devicePluginPath := resolveDevicePluginPath()
	if err := os.MkdirAll(devicePluginPath, 0755); err != nil {
		return "", err
	}

	return devicePluginPath, nil
}

func kubeletSocketPath(devicePluginPath string) string {
	if devicePluginPath == pluginapi.DevicePluginPath {
		return pluginapi.KubeletSocket
	}

	return filepath.Join(devicePluginPath, filepath.Base(pluginapi.KubeletSocket))
}

func loadConfig() (GPEmuDevicePluginConfig, error) {
	paths := []string{}
	if configPath := strings.TrimSpace(os.Getenv(ConfigFilePathEnvVar)); configPath != "" {
		paths = append(paths, configPath)
	}
	paths = append(paths, ConfigFilePath, DefaultLocalConfigFilePath)

	for _, configPath := range paths {
		raw, err := os.ReadFile(configPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return GPEmuDevicePluginConfig{}, err
		}

		var config GPEmuDevicePluginConfig
		if err := json.Unmarshal(raw, &config); err != nil {
			return GPEmuDevicePluginConfig{}, err
		}

		s, _ := json.Marshal(config)
		log.Printf("loaded config from %s: %s", configPath, string(s))
		return config, nil
	}

	log.Printf("Config file not found in %v. Using fallback config for local development.", paths)
	return GPEmuDevicePluginConfig{
		ResourceName: DefaultFallbackResourceName,
		SocketName:   "gpemu.sock",
		EGPUs:        []*EGPU{},
	}, nil
}

func main() {
	log.Println("Starging K8s eGPU Device Plugin.")

	log.Println("Starting FS watcher.")
	devicePluginPath, err := ensureDevicePluginPath()
	if err != nil {
		log.Printf("Failed to prepare device plugin path: %v", err)
		os.Exit(1)
	}

	log.Println("device plugin path: ", devicePluginPath)
	watcher, err := newFSWatcher(devicePluginPath)
	if err != nil {
		log.Println("Failed to created FS watcher.")
		os.Exit(1)
	}
	defer watcher.Close()

	log.Println("Starting OS watcher.")
	sigs := newOSWatcher(syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	config, err := loadConfig()
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	restart := true
	var devicePlugin *GPEmuDevicePlugin

L:
	for {
		if restart {
			if devicePlugin != nil {
				devicePlugin.Stop()
			}

			devicePlugin, err = NewGPEmuDevicePlugin(config, devicePluginPath)
			if err != nil {
				fmt.Println(err.Error())
				os.Exit(1)
			}
			expandedEGPUsStr := []string{}
			for _, hd := range devicePlugin.eGPUs {
				expandedEGPUsStr = append(expandedEGPUsStr, fmt.Sprintf("%+v", hd))
			}
			log.Printf("expanded egpu devices: %s\n", strings.Join(expandedEGPUsStr, ","))

			if err := devicePlugin.Serve(); err != nil {
				log.Println("Could not contact Kubelet, retrying. Did you enable the device plugin feature gate?")
			} else {
				restart = false
			}
		}

		select {
		case event := <-watcher.Events:
			if event.Name == kubeletSocketPath(devicePluginPath) && event.Op&fsnotify.Create == fsnotify.Create {
				log.Printf("inotify: %s created, restarting.", kubeletSocketPath(devicePluginPath))
				restart = true
			}

		case err := <-watcher.Errors:
			log.Printf("inotify: %s", err)

		case s := <-sigs:
			switch s {
			case syscall.SIGHUP:
				log.Println("Received SIGHUP, restarting.")
				restart = true
			default:
				log.Printf("Received signal \"%v\", shutting down.", s)
				devicePlugin.Stop()
				break L
			}
		}
	}
}

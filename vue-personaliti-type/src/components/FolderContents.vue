<template>
  <div class="container mx-auto px-6 py-6">
    <!-- Кнопка "Назад" -->
    <div class="flex items-center mb-4">
      <button
        @click="goBack"
        class="text-gray-600 hover:text-gray-800 flex items-center text-lg font-semibold"
      >
        <svg
          class="w-5 h-5 mr-1"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M15 19l-7-7 7-7"
          ></path>
        </svg>
        Назад
      </button>
    </div>

    <h1 class="text-2xl font-bold mb-6">
      Папка: {{ folderName || 'Без названия' }}
    </h1>

    <!-- Уведомление об ошибке -->
    <div
      v-if="notificationMessage"
      class="fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg z-50"
    >
      {{ notificationMessage }}
    </div>

    <!-- Секция загрузки видео -->
    <div
      class="mb-6 p-6 bg-white border border-gray-200 rounded-lg shadow-md w-full text-center"
    >
      <label class="block text-lg font-semibold text-gray-800 mb-3">
        Загрузить видео
      </label>
      <div class="flex items-center justify-center mb-4">
        <label
          v-if="!isUploading"
          class="cursor-pointer bg-blue-500 text-white font-semibold py-2 px-6 rounded-full shadow hover:bg-blue-600 transition-all duration-300"
        >
          Выберите файл
          <input
            type="file"
            @change="onFileChange"
            class="hidden"
            accept="video/*"
            multiple
            ref="fileInput"
          />
        </label>
      </div>

      <!-- Прогресс загрузки для каждого файла -->
      <div v-if="uploadingFiles.length" class="space-y-4">
        <div v-for="(file, index) in uploadingFiles" :key="index" class="mb-4">
          <p class="text-gray-600">{{ file.name }} - {{ file.size }} MB</p>
          <div
            class="relative w-64 mx-auto h-2 bg-gray-200 rounded-full overflow-hidden mt-2"
          >
            <div
              class="absolute h-full bg-blue-500 transition-all duration-300"
              :style="{ width: `${file.progress}%` }"
            ></div>
          </div>
          <p class="text-center text-gray-500 mt-2">{{ file.progress }}%</p>
        </div>
      </div>

      <p
        v-if="!isUploading && !uploadingFiles.length"
        class="mt-2 text-sm text-gray-500 text-center"
      >
        Выберите видеофайл для загрузки (поддерживаются только форматы видео).
      </p>
    </div>

    <!-- Список видео -->
    <div
      v-for="(video, index) in videos"
      :key="index"
      class="mb-4 p-4 bg-white shadow rounded-lg"
    >
      <h2
        class="text-lg font-semibold cursor-pointer text-blue-600 hover:underline"
        @click="openVideoPlayer(video.videoUrl)"
      >
        {{ video.videName }}
      </h2>
      <p class="text-gray-600">{{ video.videDescription }}</p>
      <p v-if="video.personalityType">
        Тип личности: {{ video.personalityType }}
      </p>
      <p v-if="video.suitableProfession">
        Подходящая профессия: {{ video.suitableProfession }} ({{
          video.matchPercentage
        }}%)
      </p>
    </div>

    <!-- Плеер для потоковой передачи -->
    <div
      v-if="videoPlayerUrl"
      class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
    >
      <div
        class="bg-white rounded-lg shadow-lg p-4 relative max-w-4xl w-full flex flex-col items-center"
      >
        <button
          @click="closeVideoPlayer"
          class="absolute top-4 right-6 text-gray-600 hover:text-gray-900 text-4xl font-bold z-50"
          style="line-height: 1; transform: translateY(-50%)"
        >
          &times;
        </button>
        <video
          controls
          autoplay
          :src="videoPlayerUrl"
          class="w-full max-h-[600px] rounded-lg mt-6"
        ></video>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      videos: [],
      folderName: '',
      isUploading: false,
      uploadingFiles: [],
      videoPlayerUrl: null,
      notificationMessage: '',
    }
  },
  methods: {
    goBack() {
      this.$router.go(-1)
    },

    async fetchFolderVideos() {
      const userData = JSON.parse(localStorage.getItem('user'))
      const userId = userData?.id
      const folderId = localStorage.getItem('selectedFolderId')

      if (!userId || !folderId) {
        console.error('Отсутствует user_id или folderId')
        return
      }

      try {
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/employer-card-video`,
          { user_id: userId },
          { params: { upload_id: folderId } },
        )
        this.videos = response.data.videoUrls
      } catch (error) {
        console.error('Ошибка при загрузке видео:', error)
      }
    },

    async onFileChange(event) {
      const files = Array.from(event.target.files)
      const userData = JSON.parse(localStorage.getItem('user'))
      const userId = userData.id
      const folderId = localStorage.getItem('selectedFolderId')

      if (!files.length || !folderId || !userId) return

      this.uploadingFiles = files.map(file => ({
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2),
        progress: 0,
      }))
      this.isUploading = true

      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const formData = new FormData()
        formData.append('file', file)
        formData.append('user_id', userId)
        formData.append('upload_id', folderId)

        try {
          const response = await axios.post(
            `${import.meta.env.VITE_API_URL}/employer-upload-video`,
            formData,
            {
              headers: { 'Content-Type': 'multipart/form-data' },
              onUploadProgress: progressEvent => {
                this.uploadingFiles[i].progress = Math.round(
                  (progressEvent.loaded * 100) / progressEvent.total,
                )
              },
            },
          )

          // Проверка на наличие ошибки с сообщением
          if (response.data.success === false) {
            this.showNotification(
              `Ошибка: "${file.name}" уже существует на сервере.`,
            )
          }
        } catch (error) {
          if (
            error.response &&
            error.response.data &&
            error.response.data.message
          ) {
            this.showNotification(
              `Ошибка при загрузке видео "${file.name}": ${error.response.data.message}`,
            )
          } else {
            // Общая ошибка
            console.error(`Ошибка при загрузке видео ${file.name}:`, error)
            this.showNotification(
              `Ошибка на сервере при загрузке видео "${file.name}".`,
            )
          }
        }
      }

      this.finishUploading()
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = ''
      }
    },

    finishUploading() {
      this.isUploading = false
      this.uploadingFiles = []
      this.fetchFolderVideos()
    },

    showNotification(message) {
      this.notificationMessage = message
      setTimeout(() => {
        this.notificationMessage = ''
      }, 5000)
    },

    openVideoPlayer(url) {
      this.videoPlayerUrl = `${import.meta.env.VITE_API_URL}${url}`
      window.addEventListener('keydown', this.handleEscKey)
    },

    closeVideoPlayer() {
      this.videoPlayerUrl = null
      window.removeEventListener('keydown', this.handleEscKey)
    },

    handleEscKey(event) {
      if (event.key === 'Escape') {
        this.closeVideoPlayer()
      }
    },
  },
  mounted() {
    // Получаем название папки и `folderId` из `localStorage` или параметров маршрута
    this.folderName = localStorage.getItem('selectedFolderName')
    const folderId =
      this.$route.params.folderId || localStorage.getItem('selectedFolderId')

    // Если `folderId` есть, сохраняем его в `localStorage` и загружаем данные
    if (folderId) {
      localStorage.setItem('selectedFolderId', folderId)
      this.fetchFolderVideos()
    } else {
      console.error('Отсутствует folderId для загрузки')
      this.$router.push('/') // Перенаправляем на основную страницу, если `folderId` отсутствует
    }
  },
}
</script>

<style scoped>
.container {
  max-width: 900px;
}

.video-item {
  cursor: pointer;
}

.video-player-overlay {
  background-color: rgba(0, 0, 0, 0.75);
}

.video-container {
  max-width: 900px;
}

video {
  max-height: 600px;
}

button.close-button {
  position: absolute;
  top: 16px;
  right: 16px;
  font-size: 36px;
  color: #333;
  cursor: pointer;
}
</style>

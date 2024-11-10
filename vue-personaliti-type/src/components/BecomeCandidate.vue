<template>
  <div class="container mx-auto p-6">
    <div class="bg-white rounded-lg shadow-md p-6 flex flex-col md:flex-row">
      <div class="md:w-2/3 mb-6 md:mb-0">
        <div
          class="flex items-center mb-4 bg-white border border-gray-200 rounded-lg shadow-md"
        >
          <label>
            <img
              :src="photo || defaultAvatar"
              alt="Profile Photo"
              title="Нажмите, чтобы загрузить фото"
              class="w-24 h-24 rounded-full mr-4 cursor-pointer border-2 border-gray-300 hover:border-blue-400"
              @click="uploadPhoto"
            />
            <input
              ref="photoInput"
              type="file"
              @change="onPhotoChange"
              class="hidden"
              accept="image/*"
            />
          </label>
          <div>
            <h1 class="text-2xl font-bold text-gray-800">{{ fullName }}</h1>
            <p class="text-gray-600">{{ email }}</p>
            <p class="text-gray-600">{{ phone }}</p>
          </div>
        </div>

        <!-- Секция загрузки видео -->
        <div
          class="mb-6 p-6 bg-white border border-gray-200 rounded-lg shadow-md"
        >
          <label class="block text-lg font-semibold text-gray-800 mb-3">
            Загрузить видеовизитку
          </label>

          <!-- Кнопка выбора файла -->
          <div class="flex items-center justify-center mb-4">
            <label
              v-if="!isLoadingVideo"
              class="cursor-pointer bg-blue-500 text-white font-semibold py-2 px-6 rounded-full shadow hover:bg-blue-600 transition-all duration-300"
            >
              Выберите файл
              <input
                type="file"
                @change="onFileChange"
                class="hidden"
                accept="video/*"
              />
            </label>

            <!-- Загрузка видео и кнопка отмены -->
            <div v-else class="flex items-center space-x-4">
              <div class="text-blue-500 font-semibold py-2 px-6">
                Загрузка видео...
              </div>
            </div>
          </div>

          <!-- Прогресс загрузки -->
          <div v-if="uploadingFile" class="space-y-4">
            <div class="mb-4">
              <p class="text-gray-600">
                {{ uploadingFile.name }} - {{ uploadingFile.size }} MB
              </p>
              <div
                class="relative w-64 mx-auto h-2 bg-gray-200 rounded-full overflow-hidden mt-2"
              >
                <div
                  class="absolute h-full bg-blue-500 transition-all duration-300"
                  :style="{ width: `${uploadingFile.progress}%` }"
                ></div>
              </div>
              <p class="text-center text-gray-500 mt-2">
                {{ uploadingFile.progress }}%

                <!-- Кнопка отмены -->
                <button
                  @click="cancelUpload"
                  class="block mx-auto mt-2 text-sm text-gray-500 hover:text-gray-700"
                >
                  Отменить
                </button>
              </p>
            </div>
          </div>

          <p
            v-if="!uploadingFile"
            class="mt-2 text-sm text-gray-500 text-center"
          >
            Выберите видеофайл для загрузки (поддерживаются только форматы
            видео).
          </p>
        </div>

        <!-- Видео -->
        <video
          v-if="videoUrl"
          ref="videoPlayer"
          controls
          :src="videoUrl"
          class="w-full h-auto max-w-md"
          preload="auto"
          autoplay
        />
      </div>

      <div class="md:w-1/3 md:pl-6">
        <div class="bg-gray-100 rounded-lg p-4 shadow">
          <h2 class="text-xl font-semibold text-gray-800 mb-4">
            Статистика по типу личности
          </h2>
          <p v-if="personalityStats" class="text-gray-700">
            {{ personalityStats }}
          </p>
          <p v-else class="text-gray-700">Вы еще не загрузили видео...</p>
        </div>
      </div>
    </div>

    <!-- Редактор фото -->
    <div
      v-if="isEditingPhoto"
      class="fixed inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50 z-50"
    >
      <div class="bg-white rounded-lg p-4 max-w-lg w-full">
        <h2 class="text-lg font-semibold mb-4">Редактировать фото</h2>
        <div class="relative w-full h-64">
          <img
            ref="cropperImage"
            :src="previewPhoto"
            alt="Редактируемое фото"
            class="cropper-rounded"
          />
        </div>
        <div class="flex justify-end space-x-4 mt-4">
          <button
            @click="saveCroppedPhoto"
            class="px-4 py-2 bg-blue-500 text-white rounded"
          >
            Сохранить
          </button>
          <button
            @click="cancelEditing"
            class="px-4 py-2 bg-gray-500 text-white rounded"
          >
            Отмена
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import Cropper from 'cropperjs'
import 'cropperjs/dist/cropper.css'

export default {
  data() {
    return {
      photo: '',
      defaultAvatar:
        'https://www.kindpng.com/picc/m/106-1068191_transparent-avatar-clipart-hd-png-download.png',
      fullName: '',
      email: '',
      phone: '',
      videoUrl: '',
      personalityStats: '',
      suitableVacancies: [],
      isEditingPhoto: false,
      previewPhoto: '',
      cropper: null,
      isLoadingVideo: true,
      progress: 0,
      uploadingFile: null,
      uploadCancelToken: null,
    }
  },
  methods: {
    async fetchUserData() {
      try {
        const userData = JSON.parse(localStorage.getItem('user'))
        const userId = userData.id
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/card`,
          { user_id: userId, range: this.range },
        )
        const data = response.data
        this.fullName = `${data.first_name} ${data.last_name}`
        this.email = data.login
        this.phone = data.phone
        this.personalityStats = data.personality_type
        this.suitableVacancies = data.suitableVacancies
      } catch (error) {
        console.error('Ошибка при получении данных пользователя:', error)
      }
    },

    uploadPhoto() {
      this.$refs.photoInput.click()
    },
    onPhotoChange(event) {
      const file = event.target.files[0]
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = e => {
          this.previewPhoto = e.target.result
          this.isEditingPhoto = true
        }
        reader.readAsDataURL(file)
      }
    },
    cancelEditing() {
      this.isEditingPhoto = false
      this.previewPhoto = ''
      if (this.cropper) {
        this.cropper.destroy()
      }
    },
    saveCroppedPhoto() {
      if (this.cropper) {
        this.photo = this.cropper.getCroppedCanvas().toDataURL()
        this.cancelEditing()
      }
    },
    async fetchVideo() {
      this.isLoadingVideo = true
      try {
        const userData = JSON.parse(localStorage.getItem('user'))
        if (!userData || !userData.id) {
          throw new Error('Пользователь не найден')
        }
        const userId = userData.id

        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/card-video`,
          { user_id: userId },
        )
        const videoUrl = response.data.videoUrl

        if (videoUrl.includes('undefined')) {
          console.warn('URL видео содержит "undefined", скрываю видео')
          this.videoUrl = ''
        } else {
          this.videoUrl = `${import.meta.env.VITE_API_URL}${videoUrl}`
          console.log('URL видео успешно получен:', this.videoUrl)
        }
      } catch (error) {
        console.error('Ошибка при получении видео URL:', error)
        this.videoUrl = ''
      } finally {
        this.isLoadingVideo = false
      }
    },
    onVideoLoaded() {
      const videoPlayer = this.$refs.videoPlayer
      videoPlayer.addEventListener('ended', () => {
        console.log('Видео завершено')
      })
      this.isLoadingVideo = false
    },
    async onFileChange(event) {
      const userData = JSON.parse(localStorage.getItem('user'))
      const userId = Number(userData?.id)
      const file = event.target.files[0]

      if (file && file.type.startsWith('video/')) {
        this.uploadingFile = {
          name: file.name,
          size: (file.size / (1024 * 1024)).toFixed(2),
          progress: 0,
        }
      } else {
        alert('Пожалуйста, загрузите видеофайл.')
        this.uploadingFile = null
        return
      }

      this.isLoadingVideo = true

      const formData = new FormData()
      formData.append('file', file)
      formData.append('user_id', userId)

      const CancelToken = axios.CancelToken
      this.uploadCancelToken = CancelToken.source()

      try {
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/upload-video`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: progressEvent => {
              this.uploadingFile.progress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total,
              )
            },
            cancelToken: this.uploadCancelToken.token,
          },
        )

        if (response.status === 200) {
          console.log('Видео успешно загружено')
          this.videoUrl = URL.createObjectURL(file)
        }
      } catch (error) {
        if (axios.isCancel(error)) {
          console.log('Загрузка отменена пользователем')
        } else {
          console.error('Ошибка при загрузке видео:', error)
        }
      } finally {
        this.isLoadingVideo = false
        this.uploadingFile = null
        this.uploadCancelToken = null
      }
    },

    cancelUpload() {
      if (this.uploadCancelToken) {
        this.uploadCancelToken.cancel('Отмена загрузки')
        this.isLoadingVideo = false
        this.uploadingFile = null
      }
    },
  },
  mounted() {
    this.fetchUserData()
    this.fetchVideo()
  },
  watch: {
    isEditingPhoto(newVal) {
      if (newVal) {
        this.$nextTick(() => {
          const image = this.$refs.cropperImage
          this.cropper = new Cropper(image, { aspectRatio: 1 / 1, viewMode: 1 })
        })
      }
    },
  },
}
</script>

<style scoped>
.cropper-rounded {
  border-radius: 50%;
}

.loader-bar {
  background-color: #3182ce;
  height: 100%;
  transition: width 0.3s ease;
}
</style>

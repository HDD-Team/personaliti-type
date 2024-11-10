<template>
  <div class="container w-full mx-auto px-6 py-6">
    <!-- Информация профиля -->
    <div
      class="bg-white rounded-lg shadow-md p-6 flex flex-col md:flex-row w-full justify-center text-center"
    >
      <div class="w-full">
        <div class="flex flex-col items-center mb-6">
          <label>
            <img
              :src="photo || defaultAvatar"
              alt="Profile Photo"
              title="Нажмите, чтобы загрузить фото"
              class="w-24 h-24 rounded-full cursor-pointer border-2 border-gray-300 hover:border-blue-400 mb-4"
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
          <div class="text-center">
            <h1 class="text-2xl font-bold text-gray-800">{{ fullName }}</h1>
            <p class="text-gray-600">{{ email }}</p>
            <p class="text-gray-600">{{ phone }}</p>
            <p class="text-gray-600">{{ nameCompany }}</p>
            <p class="text-gray-600">{{ overview }}</p>
          </div>
        </div>

        <!-- Секция загрузки видео с прогресс-барами для каждого файла -->
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
            <div
              v-for="(file, index) in uploadingFiles"
              :key="index"
              class="mb-4"
            >
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
            Выберите видеофайл для загрузки (поддерживаются только форматы
            видео).
          </p>
        </div>

        <!-- Сортировка -->
        <div class="flex items-center justify-end mb-4">
          <button @click="sortFolders('asc')" class="mr-2">&#9650;</button>
          <button @click="sortFolders('desc')">&#9660;</button>
        </div>

        <!-- Контейнер для папок -->
        <div
          class="folders-container grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-6 justify-center"
        >
          <div
            v-for="(folder, index) in folders"
            :key="index"
            class="folder-item bg-white border border-gray-200 shadow-md p-4 text-center"
            @click="openFolder(folder)"
          >
            <div class="folder-header">
              <div class="folder-icon"></div>
              <h2 class="folder-name">
                {{ folder.upload_name || formatDateTime(folder.upload_date) }}
              </h2>
            </div>
            <p class="text-gray-600">
              Количество видео: {{ folder.videos_url.length }}
            </p>
            <p class="text-gray-500">
              Дата загрузки: {{ formatDateTime(folder.upload_date) }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import moment from 'moment-timezone'

export default {
  data() {
    return {
      fullName: '',
      companyName: '',
      companyDescription: '',
      email: '',
      phone: '',
      nameCompany: '',
      overview: '',
      photo: '',
      defaultAvatar:
        'https://www.kindpng.com/picc/m/106-1068191_transparent-avatar-clipart-hd-png-download.png',
      isCreatingVacancy: false,
      vacancyTitle: '',
      vacancyCity: '',
      vacancyRequirements: '',
      isLoadingVideo: false,
      uploadingFiles: [],
      isUploading: false,
      folders: [],
      sortDirection: localStorage.getItem('sortDirection') || 'asc',
      folderName: '',
      folderCreatedAt: '',
      folderUpdatedAt: '',
      uploadDateTime: '',
    }
  },
  methods: {
    async fetchEmployerData() {
      try {
        const userData = JSON.parse(localStorage.getItem('user'))
        const userId = userData.id
        console.log('userId:', userId)
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/employer-card`,
          { user_id: userId },
        )
        const data = response.data

        this.fullName = `${data.first_name} ${data.last_name}`
        this.email = data.login
        this.phone = data.phone
        this.photo = data.photo
        this.defaultAvatar = data.default_avatar

        this.folders = Object.values(data.uploads).map(folder => ({
          ...folder,
          displayName:
            folder.upload_name || this.formatDateTime(folder.upload_date),
        }))

        console.log('folders:', this.folders)
      } catch (error) {
        console.error('Ошибка при получении данных работодателя:', error)
      }
    },

    uploadPhoto() {
      this.$refs.photoInput.click()
    },

    onPhotoChange(event) {
      const file = event.target.files[0]
      if (file) {
        const reader = new FileReader()
        reader.onload = e => {
          this.photo = e.target.result
        }
        reader.readAsDataURL(file)
      }
    },

    finishUploading() {
      this.isUploading = false
      this.uploadingFiles = []
      this.fetchEmployerData()
    },

    async onFileChange(event) {
      const userData = JSON.parse(localStorage.getItem('user'))
      const userId = Number(userData?.id)
      const files = Array.from(event.target.files)

      if (!files.length) return

      this.uploadDateTime = this.getCurrentDateTime()

      this.uploadingFiles = files.map(file => ({
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2), // Размер в MB
        progress: 0,
      }))

      this.isUploading = true

      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const formData = new FormData()
        formData.append('file', file)
        formData.append('user_id', userId)
        formData.append('upload_date', this.uploadDateTime)

        try {
          await axios.post(
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
        } catch (error) {
          console.error(`Ошибка при загрузке видео ${file.name}:`, error)
        }
      }

      this.finishUploading()
      this.$refs.fileInput.value = ''
    },

    updateFolderName() {
      this.folderUpdatedAt = this.getCurrentDateTime()
    },

    getCurrentDateTime() {
      const now = new Date()
      return (
        new Date(now.setHours(now.getHours() + 3)).toISOString().slice(0, -1) +
        '+03:00'
      )
    },

    formatDateTime(dateTime) {
      return moment.tz(dateTime, 'Europe/Moscow').format('YYYY-MM-DD HH:mm:ss')
    },

    sortFolders(direction) {
      this.sortDirection = direction
      this.folders = [...this.folders].sort((a, b) => {
        const dateA = moment(
          a.upload_date,
          'YYYY-MM-DD HH:mm:ss.SSSSSSZ',
        ).toDate()
        const dateB = moment(
          b.upload_date,
          'YYYY-MM-DD HH:mm:ss.SSSSSSZ',
        ).toDate()
        return direction === 'asc' ? dateA - dateB : dateB - dateA
      })
      localStorage.setItem('sortDirection', direction)
    },

    openFolder(folder) {
      if (!folder.upload_id) {
        console.error('upload_id отсутствует:', folder.upload_id)
        return
      }
      const folderId = folder.upload_id
      console.log('folderId:', folderId)
      localStorage.setItem('selectedFolderId', folder.upload_id)
      localStorage.setItem('selectedFolderName', folder.displayName)

      this.$router.push({
        name: 'folderContents',
        params: { folderId: folder.upload_id },
      })
    },
  },

  mounted() {
    this.fetchEmployerData().then(() => {
      const savedSortDirection = localStorage.getItem('sortDirection') || 'asc'
      this.sortFolders(savedSortDirection)
    })
  },
}
</script>

<style scoped>
.folder-item {
  border-radius: 8px;
  padding-top: 20px;
  overflow: hidden;
  position: relative;
}

.folder-header {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.folder-icon {
  width: 20px;
  height: 20px;
  margin-right: 8px;
  background-color: #f7ba2a;
  clip-path: polygon(0 0, 80% 0, 100% 50%, 80% 100%, 0 100%);
}

.folder-name {
  font-weight: bold;
  font-size: 1.1rem;
  color: #374151;
}

.folder-item:hover {
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}
button {
  transition:
    color 0.2s ease,
    background-color 0.2s ease;
}
button:hover {
  color: #1d4ed8;
}
</style>

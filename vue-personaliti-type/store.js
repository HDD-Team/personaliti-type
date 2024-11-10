import Vue from 'vue'
import Vuex from 'vuex'
import axios from 'axios'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    userData: {},
    videoUrl: '',
  },
  mutations: {
    SET_USER_DATA(state, data) {
      state.userData = data
    },
    SET_VIDEO_URL(state, url) {
      state.videoUrl = url
    },
  },
  actions: {
    async fetchUserData({ commit }) {
      try {
        const userData = JSON.parse(localStorage.getItem('user'))
        const userId = userData.id
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/card`,
          { user_id: userId },
        )
        const data = response.data
        commit('SET_USER_DATA', data)
      } catch (error) {
        console.error('Ошибка при получении данных пользователя:', error)
      }
    },
    async fetchVideo({ commit }) {
      try {
        const userData = JSON.parse(localStorage.getItem('user'))
        const userId = userData?.id

        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/card-video`,
          { user_id: userId },
          { responseType: 'blob' },
        )

        if (response.data.size > 0) {
          const videoBlob = new Blob([response.data], { type: 'video/mp4' })
          const videoUrl = URL.createObjectURL(videoBlob)
          commit('SET_VIDEO_URL', videoUrl)
          console.log('Видео успешно получено с сервера')
        } else {
          console.log('Видео для данного пользователя отсутствует')
          commit('SET_VIDEO_URL', '')
        }
      } catch (error) {
        console.error('Ошибка при получении видео:', error)
      }
    },
  },
})

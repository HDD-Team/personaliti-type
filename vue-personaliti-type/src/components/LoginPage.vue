<template>
  <div class="py-6">
    <div
      class="flex bg-white rounded-lg shadow-lg overflow-hidden mx-auto max-w-sm lg:max-w-4xl"
    >
      <div
        class="hidden lg:block lg:w-1/2 bg-cover"
        :style="{
          backgroundImage:
            'url(https://images.unsplash.com/photo-1546514714-df0ccc50d7bf?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=667&q=80)',
        }"
      ></div>
      <div class="w-full p-8 lg:w-1/2">
        <div class="flex items-center mb-4">
          <button @click="goBack" class="text-gray-500 hover:text-gray-700">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
        </div>
        <p class="text-xl text-gray-600 text-center">Добро пожаловать!</p>

        <div class="mt-4 flex items-center justify-between">
          <span class="border-b w-1/5 lg:w-1/4"></span>
          <span class="text-xs text-gray-500 uppercase">
            Войдите с помощью почты
          </span>
          <span class="border-b w-1/5 lg:w-1/4"></span>
        </div>

        <form @submit.prevent="login">
          <div class="mt-4">
            <label class="block text-gray-700 text-sm font-bold mb-2"
              >Email адрес</label
            >
            <input
              v-model="email"
              type="email"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-2 px-4 block w-full appearance-none"
            />
          </div>
          <div class="mt-4">
            <div class="flex justify-between">
              <label class="block text-gray-700 text-sm font-bold mb-2"
                >Пароль</label
              >
              <a href="#" class="text-xs text-gray-500">Забыли пароль?</a>
            </div>
            <input
              v-model="password"
              type="password"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-2 px-4 block w-full appearance-none"
            />
          </div>

          <div v-if="errorMessage" class="mt-2 text-red-500 text-sm">
            {{ errorMessage }}
          </div>

          <div class="mt-8">
            <button
              type="submit"
              class="bg-gray-700 text-white font-bold py-2 px-4 w-full rounded hover:bg-gray-600"
            >
              Войти
            </button>
          </div>
        </form>

        <div class="mt-4 flex items-center justify-between">
          <span class="border-b w-1/5 md:w-1/4"></span>
          <router-link to="/register" class="text-xs text-gray-500 uppercase"
            >Зарегистрироваться</router-link
          >
          <span class="border-b w-1/5 md:w-1/4"></span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      email: '',
      password: '',
      errorMessage: '',
      protectedData: null,
    }
  },
  methods: {
    async login() {
      this.errorMessage = ''

      if (!this.email || !this.password) {
        this.errorMessage = 'Пожалуйста, заполните все поля.'
        return
      }

      const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      if (!emailPattern.test(this.email)) {
        this.errorMessage = 'Неверный формат email.'
        return
      }

      if (this.password.length < 8) {
        this.errorMessage = 'Пароль должен содержать не менее 8 символов.'
        return
      }

      try {
        // Авторизация и получение токена
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/token?login=${encodeURIComponent(this.email)}&password=${encodeURIComponent(this.password)}`,
          null,
          {
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
          },
        )

        if (response.data.success) {
          const token = response.data.access_token
          localStorage.setItem('token', token)

          const userRole = response.data.role
          const userId = response.data.id
          localStorage.setItem(
            'user',
            JSON.stringify({ role: userRole, id: userId }),
          )

          // Перенаправление на главную страницу
          this.$router.push('/')
        } else {
          this.errorMessage = 'Неверный email или пароль.'
        }
      } catch (error) {
        this.errorMessage =
          'Произошла ошибка при входе. Пожалуйста, попробуйте снова.'
      }
    },
    goBack() {
      this.$router.go(-1)
    },
  },
}
</script>

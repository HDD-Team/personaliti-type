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
      <div class="w-full p-4 lg:w-1/2">
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
        <h2 class="text-xl font-semibold text-gray-700 text-center">
          Регистрация
        </h2>

        <div class="mt-2 flex items-center justify-between">
          <span class="border-b w-1/5 lg:w-1/4"></span>
          <span class="text-xs text-gray-500 uppercase text-center">
            Зарегистрируйтесь с помощью почты
          </span>
          <span class="border-b w-1/5 lg:w-1/4"></span>
        </div>

        <form @submit.prevent="register">
          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Выбрать роль:
            </label>
            <select
              v-model="role"
              class="bg-gray-200 text-gray-700 border border-gray-300 rounded py-1 px-2 block w-full"
            >
              <option value="candidate">Кандидат</option>
              <option value="employer">Работодатель</option>
            </select>
          </div>

          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Фамилия
            </label>
            <input
              v-model="lastName"
              type="text"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-1 px-2 block w-full appearance-none"
            />
          </div>

          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Имя
            </label>
            <input
              v-model="firstName"
              type="text"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-1 px-2 block w-full appearance-none"
            />
          </div>

          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Номер
            </label>
            <input
              v-model="phone"
              type="text"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-1 px-2 block w-full appearance-none"
            />
          </div>

          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Почта
            </label>
            <input
              v-model="email"
              type="email"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-1 px-2 block w-full appearance-none"
            />
          </div>

          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Пароль
            </label>
            <input
              v-model="password"
              type="password"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-1 px-2 block w-full appearance-none"
            />
          </div>

          <div class="mt-2">
            <label class="block text-gray-700 text-sm font-bold mb-1">
              Подтвердить пароль
            </label>
            <input
              v-model="confirmPassword"
              type="password"
              required
              class="bg-gray-200 text-gray-700 focus:outline-none focus:shadow-outline border border-gray-300 rounded py-1 px-2 block w-full appearance-none"
            />
          </div>

          <div v-if="errorMessage" class="mt-2 text-red-500 text-sm">
            {{ errorMessage }}
          </div>

          <div v-if="successMessage" class="mt-2 text-green-500 text-sm">
            {{ successMessage }}
          </div>

          <div class="mt-4">
            <button
              type="submit"
              class="bg-gray-700 text-white font-bold py-1 px-2 w-full rounded hover:bg-gray-600"
            >
              Зарегистрироваться
            </button>
          </div>
        </form>

        <div class="mt-2 flex items-center justify-between">
          <span class="border-b w-1/5 md:w-1/4"></span>
          <router-link to="/login" class="text-xs text-gray-500 uppercase">
            Войти
          </router-link>
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
      role: 'candidate', // По умолчанию
      lastName: 'Далиба',
      firstName: 'богдан',
      phone: '+79788001972',
      email: 'bogdandaliba@gmail.com',
      password: '124345678',
      confirmPassword: '12345678',
      errorMessage: '',
      successMessage: '',
    }
  },
  methods: {
    goBack() {
      this.$router.go(-1)
    },
    async register() {
      this.errorMessage = ''
      this.successMessage = ''

      if (
        !this.email ||
        !this.password ||
        !this.confirmPassword ||
        !this.lastName ||
        !this.firstName ||
        !this.phone
      ) {
        this.errorMessage = 'Пожалуйста, заполните все поля.'
        return
      }

      if (this.password !== this.confirmPassword) {
        this.errorMessage = 'Пароли не совпадают.'
        return
      }

      if (this.password.length < 8) {
        this.errorMessage = 'Пароль должен содержать не менее 8 символов.'
        return
      }

      const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      if (!emailPattern.test(this.email)) {
        this.errorMessage = 'Неверный формат email.'
        return
      }

      try {
        const response = await axios.post(
          `${import.meta.env.VITE_API_URL}/register`,
          {
            role: this.role,
            last_name: this.lastName,
            first_name: this.firstName,
            phone: this.phone,
            login: this.email,
            password: this.password,
          },
        )

        if (response.data.success) {
          this.successMessage = 'Вы успешно зарегистрированы.'
          setTimeout(() => {
            this.$router.push('/login')
          }, 2000)
        } else {
          this.errorMessage =
            response.data.message ||
            'Произошла ошибка при регистрации. Пожалуйста, попробуйте снова.'
        }
      } catch (error) {
        // Проверка на сообщение об уже существующем email
        if (
          error.response &&
          error.response.data &&
          error.response.data.detail.message === 'Email already exists.'
        ) {
          this.errorMessage = 'Этот email уже зарегистрирован.'
        } else {
          this.errorMessage =
            'Произошла ошибка при регистрации. Пожалуйста, попробуйте снова.'
        }
      }
    },
  },
}
</script>

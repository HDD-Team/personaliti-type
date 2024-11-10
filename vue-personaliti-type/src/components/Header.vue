<template>
  <div class="bg-white border py-3 px-6">
    <div class="flex justify-between items-center">
      <!-- логотип и название -->
      <div class="flex items-center">
        <img src="@/assets/HDD_logo.jpg" alt="Логотип" class="h-10 w-10 mr-2" />
        <span class="font-semibold text-[#252C32] text-lg">HDD</span>
      </div>

      <!-- Кнопки авторизации -->
      <div class="ml-2 flex">
        <!-- Если пользователь не авторизован -->
        <div v-if="!isLoggedIn" class="flex space-x-4">
          <button
            class="ml-2 flex items-center gap-x-1 rounded-md border py-2 px-4 hover:bg-gray-100"
            @click="goToLogin"
          >
            <span class="text-sm font-medium">Войти</span>
          </button>
        </div>

        <!-- Если пользователь авторизован -->
        <div v-else class="flex items-center gap-x-4">
          <div v-if="userRole === 'employer'" class="flex items-center gap-x-1">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-5 w-5 text-gray-500"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path d="M4 3a2 2 0 100 4h12a2 2 0 100-4H4z" />
              <path
                fill-rule="evenodd"
                d="M3 8h14v7a2 2 0 01-2 2H5a2 2 0 01-2-2V8zm5 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"
                clip-rule="evenodd"
              />
            </svg>
            <span class="text-sm font-medium">Аккаунт работодателя</span>
          </div>
          <div v-else class="flex items-center gap-x-1">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-5 w-5 text-gray-500"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fill-rule="evenodd"
                d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z"
                clip-rule="evenodd"
              />
            </svg>
            <span class="text-sm font-medium">Аккаунт кандидата</span>
          </div>
          <button
            class="ml-2 flex items-center gap-x-1 rounded-md border py-2 px-4 hover:bg-gray-100"
            @click="logout"
          >
            <span class="text-sm font-medium">Выйти</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      isLoggedIn: false,
      userRole: null,
    }
  },
  methods: {
    goToLogin() {
      this.$router.push('/login')
    },

    logout() {
      localStorage.removeItem('user')
      this.isLoggedIn = false
      this.userRole = null
      this.$router.push('/login')
    },
  },
  mounted() {
    const user = JSON.parse(localStorage.getItem('user'))
    if (user) {
      this.isLoggedIn = true
      this.userRole = user.role
    }
  },
}
</script>

<style scoped>
button {
  transition:
    color 0.2s ease,
    background-color 0.2s ease;
}
button:hover {
  color: #1d4ed8;
}
</style>

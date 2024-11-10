<template>
  <div>
    <Header />
    <router-view />
  </div>
</template>

<script>
import Header from './Header.vue'

export default {
  components: { Header },
  beforeCreate() {
    const user = JSON.parse(localStorage.getItem('user'))
    if (user) {
      if (user.role === 'candidate') {
        this.$router.push('/become-candidate')
      } else if (user.role === 'employer') {
        this.$router.push('/become-employer')
      }
    } else {
      this.$router.push('/login')
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

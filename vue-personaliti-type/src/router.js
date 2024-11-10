import { createRouter, createWebHistory } from 'vue-router'
import LoginPage from '@/components/LoginPage.vue'
import MainLayout from '@/components/MainLayout.vue'
import BecomeCandidate from '@/components/BecomeCandidate.vue'
import BecomeEmployer from '@/components/BecomeEmployer.vue'
import RegisterPage from '@/components/RegisterPage.vue'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: LoginPage,
  },
  {
    path: '/register',
    name: 'Register',
    component: RegisterPage,
  },
  {
    component: MainLayout, // Используем основной компонент для маршрутов
    children: [
      {
        path: '/become-candidate', // Добавляем маршрут для формы кандидата
        name: 'BecomeCandidate',
        component: BecomeCandidate, // Можно оставить так или использовать динамический импорт
      },
      {
        path: '/become-employer', // Добавляем маршрут для формы кандидата
        name: 'BecomeEmployer',
        component: BecomeEmployer, // Можно оставить так или использовать динамический импорт
      },
      {
        path: '/folder/:folderId',
        name: 'folderContents',
        component: () => import('@/components/FolderContents.vue'),
        props: true,
        meta: { requiresAuth: true }, // Убедитесь, что здесь включена защита
      },
    ],
  },
  {
    path: '/:catchAll(.*)',
    redirect: '/login',
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router

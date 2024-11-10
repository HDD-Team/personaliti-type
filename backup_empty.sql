--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3
-- Dumped by pg_dump version 16.3

-- Started on 2024-11-10 00:15:04

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 224 (class 1255 OID 33010)
-- Name: set_login_condidates(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.set_login_condidates() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.login := (SELECT login FROM users WHERE user_id = NEW.candidate_id);
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.set_login_condidates() OWNER TO postgres;

--
-- TOC entry 223 (class 1255 OID 33011)
-- Name: set_login_emloyers(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.set_login_emloyers() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.login := (SELECT login FROM users WHERE user_id = NEW.employer_id);
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.set_login_emloyers() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 218 (class 1259 OID 32840)
-- Name: candidates; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.candidates (
    id integer NOT NULL,
    candidate_id integer,
    login character varying(50) NOT NULL,
    first_name character varying(50) NOT NULL,
    last_name character varying(50) NOT NULL,
    phone character varying(20),
    profile_video_url character varying(255),
    personality_type character varying(50),
    profile_photo_url character varying(255)
);


ALTER TABLE public.candidates OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 32839)
-- Name: candidates_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.candidates_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.candidates_id_seq OWNER TO postgres;

--
-- TOC entry 4892 (class 0 OID 0)
-- Dependencies: 217
-- Name: candidates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.candidates_id_seq OWNED BY public.candidates.id;


--
-- TOC entry 220 (class 1259 OID 32980)
-- Name: employers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.employers (
    id integer NOT NULL,
    employer_id integer,
    login character varying(50) NOT NULL,
    first_name character varying(50) NOT NULL,
    last_name character varying(50) NOT NULL,
    phone character varying(20),
    company_name character varying(100),
    company_description text
);


ALTER TABLE public.employers OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 32979)
-- Name: employers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.employers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.employers_id_seq OWNER TO postgres;

--
-- TOC entry 4893 (class 0 OID 0)
-- Dependencies: 219
-- Name: employers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.employers_id_seq OWNED BY public.employers.id;


--
-- TOC entry 222 (class 1259 OID 33201)
-- Name: employers_videos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.employers_videos (
    id integer NOT NULL,
    employer_id integer NOT NULL,
    upload_id integer NOT NULL,
    videos_url text[] NOT NULL,
    upload_date timestamp with time zone NOT NULL,
    upload_name text,
    video_descriptions text[]
);


ALTER TABLE public.employers_videos OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 33200)
-- Name: employers_videos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.employers_videos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.employers_videos_id_seq OWNER TO postgres;

--
-- TOC entry 4894 (class 0 OID 0)
-- Dependencies: 221
-- Name: employers_videos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.employers_videos_id_seq OWNED BY public.employers_videos.id;


--
-- TOC entry 216 (class 1259 OID 32784)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    user_id integer NOT NULL,
    login character varying(50) NOT NULL,
    password character varying(100) NOT NULL,
    role character varying(20) NOT NULL
);


ALTER TABLE public.users OWNER TO postgres;

--
-- TOC entry 215 (class 1259 OID 32783)
-- Name: users_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_user_id_seq OWNER TO postgres;

--
-- TOC entry 4895 (class 0 OID 0)
-- Dependencies: 215
-- Name: users_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_user_id_seq OWNED BY public.users.user_id;


--
-- TOC entry 4706 (class 2604 OID 32843)
-- Name: candidates id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates ALTER COLUMN id SET DEFAULT nextval('public.candidates_id_seq'::regclass);


--
-- TOC entry 4707 (class 2604 OID 32983)
-- Name: employers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers ALTER COLUMN id SET DEFAULT nextval('public.employers_id_seq'::regclass);


--
-- TOC entry 4708 (class 2604 OID 33204)
-- Name: employers_videos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers_videos ALTER COLUMN id SET DEFAULT nextval('public.employers_videos_id_seq'::regclass);


--
-- TOC entry 4705 (class 2604 OID 32787)
-- Name: users user_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN user_id SET DEFAULT nextval('public.users_user_id_seq'::regclass);


--
-- TOC entry 4882 (class 0 OID 32840)
-- Dependencies: 218
-- Data for Name: candidates; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.candidates (id, candidate_id, login, first_name, last_name, phone, profile_video_url, personality_type, profile_photo_url) FROM stdin;
\.


--
-- TOC entry 4884 (class 0 OID 32980)
-- Dependencies: 220
-- Data for Name: employers; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.employers (id, employer_id, login, first_name, last_name, phone, company_name, company_description) FROM stdin;
\.


--
-- TOC entry 4886 (class 0 OID 33201)
-- Dependencies: 222
-- Data for Name: employers_videos; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.employers_videos (id, employer_id, upload_id, videos_url, upload_date, upload_name, video_descriptions) FROM stdin;
\.


--
-- TOC entry 4880 (class 0 OID 32784)
-- Dependencies: 216
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.users (user_id, login, password, role) FROM stdin;
\.


--
-- TOC entry 4896 (class 0 OID 0)
-- Dependencies: 217
-- Name: candidates_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.candidates_id_seq', 2, true);


--
-- TOC entry 4897 (class 0 OID 0)
-- Dependencies: 219
-- Name: employers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.employers_id_seq', 1, true);


--
-- TOC entry 4898 (class 0 OID 0)
-- Dependencies: 221
-- Name: employers_videos_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.employers_videos_id_seq', 2, true);


--
-- TOC entry 4899 (class 0 OID 0)
-- Dependencies: 215
-- Name: users_user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.users_user_id_seq', 3, true);


--
-- TOC entry 4714 (class 2606 OID 32847)
-- Name: candidates candidates_candidate_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates
    ADD CONSTRAINT candidates_candidate_id_key UNIQUE (candidate_id);


--
-- TOC entry 4716 (class 2606 OID 32849)
-- Name: candidates candidates_login_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates
    ADD CONSTRAINT candidates_login_key UNIQUE (login);


--
-- TOC entry 4718 (class 2606 OID 32845)
-- Name: candidates candidates_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates
    ADD CONSTRAINT candidates_pkey PRIMARY KEY (id);


--
-- TOC entry 4720 (class 2606 OID 32989)
-- Name: employers employers_emloyer_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers
    ADD CONSTRAINT employers_emloyer_id_key UNIQUE (employer_id);


--
-- TOC entry 4722 (class 2606 OID 32991)
-- Name: employers employers_login_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers
    ADD CONSTRAINT employers_login_key UNIQUE (login);


--
-- TOC entry 4724 (class 2606 OID 32987)
-- Name: employers employers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers
    ADD CONSTRAINT employers_pkey PRIMARY KEY (id);


--
-- TOC entry 4726 (class 2606 OID 33208)
-- Name: employers_videos employers_videos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers_videos
    ADD CONSTRAINT employers_videos_pkey PRIMARY KEY (id);


--
-- TOC entry 4710 (class 2606 OID 32792)
-- Name: users users_login_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_login_key UNIQUE (login);


--
-- TOC entry 4712 (class 2606 OID 32790)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (user_id);


--
-- TOC entry 4734 (class 2620 OID 33012)
-- Name: candidates set_login_candidates_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER set_login_candidates_trigger BEFORE INSERT ON public.candidates FOR EACH ROW EXECUTE FUNCTION public.set_login_condidates();


--
-- TOC entry 4735 (class 2620 OID 33013)
-- Name: employers set_login_emloyers_trigger; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER set_login_emloyers_trigger BEFORE INSERT ON public.employers FOR EACH ROW EXECUTE FUNCTION public.set_login_emloyers();


--
-- TOC entry 4727 (class 2606 OID 32850)
-- Name: candidates candidates_candidate_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates
    ADD CONSTRAINT candidates_candidate_id_fkey FOREIGN KEY (candidate_id) REFERENCES public.users(user_id) ON DELETE CASCADE;


--
-- TOC entry 4728 (class 2606 OID 32855)
-- Name: candidates candidates_candidate_id_fkey1; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates
    ADD CONSTRAINT candidates_candidate_id_fkey1 FOREIGN KEY (candidate_id) REFERENCES public.users(user_id);


--
-- TOC entry 4729 (class 2606 OID 32860)
-- Name: candidates candidates_login_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.candidates
    ADD CONSTRAINT candidates_login_fkey FOREIGN KEY (login) REFERENCES public.users(login);


--
-- TOC entry 4730 (class 2606 OID 32992)
-- Name: employers employers_emloyer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers
    ADD CONSTRAINT employers_emloyer_id_fkey FOREIGN KEY (employer_id) REFERENCES public.users(user_id) ON DELETE CASCADE;


--
-- TOC entry 4731 (class 2606 OID 32997)
-- Name: employers employers_emloyer_id_fkey1; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers
    ADD CONSTRAINT employers_emloyer_id_fkey1 FOREIGN KEY (employer_id) REFERENCES public.users(user_id);


--
-- TOC entry 4732 (class 2606 OID 33002)
-- Name: employers employers_login_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers
    ADD CONSTRAINT employers_login_fkey FOREIGN KEY (login) REFERENCES public.users(login);


--
-- TOC entry 4733 (class 2606 OID 33209)
-- Name: employers_videos employers_videos_employer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.employers_videos
    ADD CONSTRAINT employers_videos_employer_id_fkey FOREIGN KEY (employer_id) REFERENCES public.employers(employer_id);


-- Completed on 2024-11-10 00:15:04

--
-- PostgreSQL database dump complete
--


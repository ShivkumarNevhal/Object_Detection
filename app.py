# -------------------------------
# RIGHT SIDE (Detection)
# -------------------------------
with col2:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("🧠 Detection Result")

    input_img = None

    # Unified input source
    if uploaded_file:
        input_img = Image.open(uploaded_file)

    elif "camera_image" in st.session_state:
        input_img = st.session_state["camera_image"]

    if input_img:

        if st.button("🔍 Run Detection"):

            # -------------------------------
            # ✅ STANDARD INPUT PIPELINE
            # -------------------------------

            # Convert to RGB (safety)
            input_img = input_img.convert("RGB")

            # Convert to numpy
            img_np = np.array(input_img)

            # Resize (VERY IMPORTANT)
            img_np = cv2.resize(img_np, (640, 640))

            # Optional: reduce noise (helps camera input)
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

            # -------------------------------
            # Run Model
            # -------------------------------
            results = model(img_np, conf=0.6)

            # Convert result for display
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # -------------------------------
            # SHOW INPUT + OUTPUT
            # -------------------------------
            colA, colB = st.columns(2)

            with colA:
                st.image(input_img, caption="Input Image")

            with colB:
                st.image(result_img, caption="Detected Image")

            # -------------------------------
            # CLEAN RESULTS (NO HALLUCINATION)
            # -------------------------------
            st.subheader("Results")

            if len(results[0].boxes) == 0:
                st.warning("No objects detected 🚫")

            else:
                # Only show best prediction
                best_box = max(results[0].boxes, key=lambda x: float(x.conf[0]))

                conf = float(best_box.conf[0])
                cls = int(best_box.cls[0])

                if conf > 0.6:
                    st.success(f"{model.names[cls]} → {conf:.2f}")
                else:
                    st.warning("Low confidence detection ⚠️")

    st.markdown('</div>', unsafe_allow_html=True)

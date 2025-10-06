import { supabase } from "../lib/supabaseClient";

const BUCKET =
  (typeof process !== "undefined" && process.env?.REACT_APP_SB_BUCKET) ||
  "incident-photos";

// Set to "true" (string) in env if your bucket is PRIVATE and you want signed URLs
const USE_SIGNED_URLS =
  (typeof process !== "undefined" && process.env?.REACT_APP_SB_SIGNED) ===
  "true";

// Sanitize filenames
function cleanName(name = "") {
  return name.replace(/[^\w.\-]+/g, "_").slice(0, 120);
}

export async function saveIncidentReport({ form, user }) {
  let photoUrl = null;

  // ---- 1) Optional photo upload ----
  if (form?.photo instanceof File) {
    const userId = user?.id || "guest";
    const fname = `${Date.now()}-${cleanName(form.photo.name || "photo.jpg")}`;
    const key = `${userId}/${fname}`;

    const { data: up, error: upErr } = await supabase.storage
      .from(BUCKET)
      .upload(key, form.photo, { cacheControl: "3600", upsert: false });

    if (upErr) {
      if (/Bucket not found/i.test(upErr.message)) {
        throw new Error(
          `Storage bucket "${BUCKET}" not found. Create it in Supabase → Storage (or set REACT_APP_SB_BUCKET to an existing bucket).`
        );
      }
      throw upErr;
    }

    if (USE_SIGNED_URLS) {
      // PRIVATE bucket
      const { data: signed, error } = await supabase.storage
        .from(BUCKET)
        .createSignedUrl(up.path, 60 * 60 * 24 * 30); // 30 days
      if (error) throw error;
      photoUrl = signed?.signedUrl ?? null;
    } else {
      // PUBLIC bucket
      const { data: pub } = supabase.storage.from(BUCKET).getPublicUrl(up.path);
      photoUrl = pub?.publicUrl ?? null;
    }
  }

  // ---- 2) Map form → DB row ----
  const payload = {
    user: user?.name || user?.email || "Guest",
    incidentType: form?.type || null,
    severity: (form?.severity || "").toLowerCase(),
    location:
      form?.location?.road ||
      (form?.location?.gps
        ? `${form.location.gps.lat},${form.location.gps.lng}`
        : null),
    fullAddress: form?.location?.fullAddress || null,
    description: form?.description || null,
    photo_url: photoUrl,
    createdAt: new Date().toISOString(),
  };

  const { data, error } = await supabase
    .from("incident_report")
    .insert([payload])
    .select()
    .single();

  if (error) throw error;
  return data;
}

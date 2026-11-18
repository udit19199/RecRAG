Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=b900c8e46f4500a4 ===
Gold sentences and facts:
Sentence 0: Duncan 's diagnosis was publicly confirmed during a CDC news conference the same day . That evening , Duncan reported feeling better and requested to watch a movie . The following morning , Duncan was breathing rapidly and complaining of `` pain all over '' . By the afternoon , however , he was able to eat , and the doctor noted that he was feeling better . The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola virus disease and the index patient for the later infections of nurses Pham and Vinson .
  Fact 0.0: The CDC confirmed Duncan's Ebola diagnosis at a news conference on the same day.
  Fact 0.1: That evening, Duncan reported feeling better and requested to watch a movie.
  Fact 0.2: The following morning, Duncan was breathing rapidly and complaining of 'pain all over'.
  Fact 0.3: By the afternoon, Duncan was able to eat.
  Fact 0.4: By the afternoon, the doctor noted that Duncan was feeling better.
  Fact 0.5: On October 3, Duncan again reported feeling abdominal pain.
  Fact 0.6: That evening, October 3, the hospital contacted Chimerix.
  Fact 0.7: Chimerix is a biotechnology company developing Brincidofovir to combat the disease.
  Fact 0.8: On October 4, Duncan's organs were failing.
  Fact 0.9: On October 4, Duncan was intubated to help him breathe.
  Fact 0.10: On October 4 in the afternoon, the hospital began administering Brincidofovir to Duncan.
  Fact 0.11: Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock.
  Fact 0.12: On October 7, the hospital reported that Duncan's condition was improving.
  Fact 0.13: Duncan died at 7:51 a.m. on October 8.
  Fact 0.14: Duncan became the first person to die in the United States of Ebola virus disease.
  Fact 0.15: Duncan was the index patient for the later infections of nurses Pham and Vinson.
Sentence 1: In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola '' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first was reported in September 2014 . Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered .
  Fact 1.0: In December 2014, Ebola virus cases occurred in the United States due to four laboratory-confirmed cases of Ebola virus disease in the United States.
  Fact 1.1: Ebola virus disease is commonly known as Ebola.
  Fact 1.2: Eleven Ebola cases have been reported.
  Fact 1.3: The eleven reported cases include the four cases diagnosed in the United States and seven cases medically evacuated from other countries.
  Fact 1.4: The first of the eleven cases was reported in September 2014.
  Fact 1.5: Nine of the eleven cases contracted the disease outside the US and traveled into the country.
  Fact 1.6: The nine people traveled into the country either as regular airline passengers or as medical evacuees.
  Fact 1.7: Two of those nine people died.
  Fact 1.8: Two people have contracted Ebola in the United States.
  Fact 1.9: Both people who contracted Ebola in the United States were nurses who treated an Ebola patient.
  Fact 1.10: Both nurses have recovered.
Sentence 2: On September 30 , 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 , Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were :
  Fact 2.0: The CDC announced on September 30, 2014 that Thomas Eric Duncan had been diagnosed with Ebola in Dallas, Texas.
  Fact 2.1: Thomas Eric Duncan was reportedly 42 years old.
  Fact 2.2: CDC reports later corrected Thomas Eric Duncan's age to 45 years old.
  Fact 2.3: Thomas Eric Duncan was a Liberian national.
  Fact 2.4: Thomas Eric Duncan was visiting the United States from Liberia.
  Fact 2.5: Thomas Eric Duncan had been visiting family in Dallas.
  Fact 2.6: Thomas Eric Duncan was treated at Texas Health Presbyterian Hospital Dallas.
  Fact 2.7: By October 4, Thomas Eric Duncan's condition had deteriorated from 'serious but stable' to 'critical'.
  Fact 2.8: Thomas Eric Duncan died of Ebola on October 8.
  Fact 2.9: Three other cases were diagnosed in the United States as of October 2014, and they are listed after this sentence.
Sentence 3: Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
  Fact 3.0: The page is about Ebola virus disease in the U.S.
  Fact 3.1: The page includes a map of Ebola cases and infrastructure throughout the U.S.
  Fact 3.2: The page has categories for cases contracted in the U.S., cases first diagnosed in the U.S., and cases evacuated to the U.S. from other countries.
  Fact 3.3: The table lists 7 cases evacuated to the U.S. from other countries.
  Fact 3.4: The table lists 11 total cases.
  Fact 3.5: The table has a 'Deaths' label.
  Fact 3.6: The table lists 9 recoveries from Ebola.
  Fact 3.7: The table lists 0 active cases.

Retrieved items:
<item 1>
Ebola virus cases in the United States
Medical evacuations from West Africa ( edit ) Local transmissions -- no deaths : Texas Initial cases -- deaths : Texas Initial cases -- no deaths : New York
Medically evacuated cases -- no deaths : Georgia , Maryland Medically evacuated cases -- deaths : Nebraska By the time the Ebola virus epidemic in West Africa
was over in June 2016 , seven Americans had been evacuated to the U.S. for treatment after contracting Ebola virus while working with medical teams trying to
stop the epidemic . Kent Brantly , a physician and medical director in Liberia for the aid group Samaritan 's Purse , and co-worker Nancy Writebol were infected
in July 2014 , while working in Monrovia . Both were flown to the United States at the beginning of August via the CDC 's Aeromedical Biological Containment
System for further treatment in Atlanta 's Emory University Hospital . On August 21 , Brantly and Writebol recovered and were discharged . Brantly has since
donated blood to three others with Ebola ( Sacra , Mukpo and Pham ) .
</item 1>

<item 2>
Ebola virus cases in the United States
( edit ) Articles related to the West African Ebola virus epidemic Overview Ebola virus disease Timeline of the epidemic Responses to the epidemic Timeline ,
tables and graphs Ebola virus disease treatment research Post-Ebola virus syndrome Ebola vaccine Nations with widespread cases Guinea Liberia Sierra Leone Other
affected nations Mali Nigeria Senegal Spain United States United Kingdom Italy Other outbreaks List of Ebola outbreaks 1976 Zaire outbreak 2014 DR Congo
outbreak List of epidemics First case : Thomas Eric Duncan ( edit ) Main article : Thomas Eric Duncan Thomas Eric Duncan in Liberia ( edit ) Thomas Eric Duncan
was from Monrovia , Liberia , to date the country hit hardest by the Ebola virus epidemic . Duncan worked as a personal driver for the general manager of
Safeway Cargo , a FedEx contractor in Liberia . According to manager Henry Brunson , Duncan had abruptly quit his job on September 4 , 2014 , giving no reason .
On September 15 , 2014 , the family of Marthalene Williams , who later died of Ebola virus disease , could not call an ambulance to transfer the pregnant
Williams to a hospital . Duncan , their tenant , helped to transfer Williams by taxi to an Ebola treatment ward in Monrovia .
</item 2>

<item 3>
Ebola virus cases in the United States
Ebola virus cases in the United States - wikipedia Ebola virus cases in the United States Jump to : navigation , search This article is about Ebola virus cases
in the United States starting in 2014 . For the 2013 - 2016 outbreak in Africa , see Ebola virus epidemic in West Africa . For the U.S. discovery of a different
species of the Ebolavirus genus in 1989 , see Reston virus . Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases
contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola
'' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first
was reported in September 2014 .
</item 3>

<item 4>
Ebola virus cases in the United States
See also ( edit ) Ebola virus epidemic in Guinea Ebola virus epidemic in Liberia Ebola virus epidemic in Nigeria Ebola virus epidemic in Sierra Leone Ebola
virus disease in Spain List of Ebola outbreaks West Nile virus in the United States Notes ( edit ) Jump up ^ Duncan contracted Ebola in Liberia and was
diagnosed in the U.S. Two of the victims had had direct contact with Duncan . One victim was diagnosed in New York City ; his case is unrelated to the Texas
cases . ^ Jump up to : This takes into account the seven Americans who were medically evacuated from West Africa with Ebola . References ( edit ) Jump up ^ ``
Who are the American Ebola patients ? '' , CNN , October 6 , 2014 Jump up ^ `` Nebraska Hospital : Surgeon With Ebola Has Died '' . ABC News . November 17 ,
2014 . Retrieved November 17 , 2014 . Jump up ^ `` 2014 Ebola Outbreak - Case Count '' . CDC . December 15 , 2014 . Retrieved December 16 , 2014 . Jump up ^ ``
First diagnosed case of Ebola in the U.S. '' CNN .
</item 4>

<item 5>
Ebola virus cases in the United States
CBS News . November 13 , 2014 . Jump up ^ Sieff , Kevin ( November 17 , 2014 ) . `` A doctor 's mistaken Ebola test : ' We were celebrating ... Then everything
fell apart ' '' . Washington Post . Jump up ^ `` Martin Salia , Surgeon With Ebola , Arrives in Nebraska From Sierra Leone '' . New York Times . November 15 ,
2014 . Retrieved November 16 , 2014 . Jump up ^ Goodnough , Abby ; Trenchard , Tommy ( November 17 , 2014 ) . `` Doctor Being Treated for Ebola in Omaha Dies ''
. The New York Times . Jump up ^ U.S.
</item 5>

<item 6>
Ebola virus cases in the United States
Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those
nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered . On September 30
, 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national
visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at
Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 ,
Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were : October 11 , 2014 , a nurse , Nina Pham , who had provided
care to Duncan at the hospital . October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan .
</item 6>

<item 7>
Ebola virus cases in the United States
Jump up ^ Selk , Avi ( October 8 , 2014 ) . `` Before his death , Duncan said he mistook Ebola case in Liberia for miscarriage ; never lied '' . Dallas News .
Jump up ^ `` Ebola Outbreak and Outcry : Saving Thomas Eric Duncan '' . The Huffington Post . October 7 , 2014 . Retrieved October 17 , 2014 . Jump up ^ ``
Family of Thomas Eric Duncan says his death from Ebola is ' racially motivated ' '' . WGN - TV . Retrieved October 17 , 2014 . Jump up ^ `` Dallas ' Ebola
patient waited days for experimental drug '' . CNN . October 7 , 2014 . Retrieved October 17 , 2014 . Jump up ^ Ebola Victim Thomas Eric Duncan 's 9 - Day
Treatment Cost Hospital Estimated $500 K and He Had No Insurance , ChristianPost.com , October 10 , 2014 ; accessed October 27 , 2014 . Jump up ^ ABC News . ``
Dallas Hospital ' Feels Like a Ghost Town ' in Wake of Ebola Cases '' . ABC News .
</item 7>

<item 8>
Ebola virus cases in the United States
October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan . October 23 , 2014 , physician Craig Spencer , diagnosed in New York City ; he had just
returned from working with Doctors Without Borders in Guinea , a country in west Africa . Hundreds of people were tested or monitored for potential Ebola virus
infection , but the two nurses were the only confirmed cases of locally transmitted Ebola . Public health experts and the Obama administration opposed
instituting a travel ban on Ebola endemic areas , stating that it would be ineffective and would paradoxically worsen the situation . No one both contracted and
died of Ebola virus disease while in the United States . No new cases have been diagnosed in the United States since Craig Spencer was released from a New York
City hospital on November 11 , 2014 . Contents ( hide ) 1 Cases diagnosed in U.S. 1.1 First case : Thomas Eric Duncan 1.1. 1 Thomas Eric Duncan in Liberia 1.1.
2 Duncan 's illness in Dallas 1.1. 3 Contact tracing 1.1.
</item 8>

<item 9>
Ebola virus cases in the United States
Duncan rode in the taxi to the treatment ward with Williams , her father and her brother . On September 19 , Duncan went to Monrovia Airport , where , according
to Liberian officials , Duncan lied about his history of contact with the disease on an airport questionnaire before boarding a Brussels Airlines flight to
Brussels . In Brussels , Duncan boarded United Airlines Flight 951 to Washington Dulles Airport . From Dulles , he boarded United Airlines Flight 822 to Dallas
/ Fort Worth . He arrived in Dallas at 7 : 01 p.m. CDT on September 20 , 2014 , and stayed with his partner and her five children , who lived in the Fair Oaks
neighborhood of Dallas . Vickery Meadow , the region of Dallas where Duncan resided , has a large African immigrant population and is Dallas 's densest
neighborhood . Duncan 's illness in Dallas ( edit ) Texas Presbyterian Hospital , where Duncan was treated Duncan began experiencing symptoms on September 24 ,
2014 , and arrived at the Texas Health Presbyterian Hospital emergency room at 10 : 37 p.m. on September 25 . At 11 : 36 p.m.
</item 9>

<item 10>
Ebola virus cases in the United States
The doctor , identified later as Dr. Martin Salia , became symptomatic while in Sierra Leone . His initial Ebola test came back negative , but his symptoms
persisted . He attempted to treat his symptoms , which included vomiting and diarrhea , believing he had malaria . A second test for Ebola came back positive .
His family have expressed concern that the delay in diagnosis might impact his recovery . Dr. Salia arrived at Eppley Airfield in Omaha on November 15 , and was
transported to the Nebraska Medical Center . According to the team that assisted in the transport , he was critically ill and considered to be the sickest
patient to be evacuated , but stable enough to fly . On November 17 , Dr. Salia died from the disease . Dr. Philip Smith , medical director of the
biocontainment unit at Nebraska Medical Center , said Salia 's disease was already `` extremely advanced '' by the time he arrived in Omaha . By then , Salia 's
kidneys had failed and he was in acute respiratory distress . Salia 's treatment included a blood plasma transfusion from an Ebola survivor , as well as the
experimental drug ZMapp .
</item 10>
=== END ITEM id=b900c8e46f4500a4 ===

=== ITEM id=23abe963ec738b9e ===
Gold sentences and facts:
Sentence 0: The uvea is the vascular middle layer of the eye . It is traditionally divided into three areas , from front to back , the :
  Fact 0.0: The uvea is the vascular middle layer of the eye.
  Fact 0.1: The uvea is traditionally divided into three areas.
  Fact 0.2: The three areas of the uvea are divided from front to back.

Retrieved items:
<item 1>
Retina
Function , and Evolution '' Retina -- Cell Centered Database Histology image : 07901loa -- Histology Learning System at Boston University MedlinePlus
Encyclopedia 002291 ( hide ) Anatomy of the globe of the human eye Fibrous tunic ( outer ) Sclera Episcleral layer Schlemm 's canal Trabecular meshwork Cornea
Limbus layers Epithelium Bowman 's Stroma Dua 's layer Descemet 's Endothelium Uvea / vascular tunic ( middle ) Choroid Capillary lamina of choroid Bruch 's
membrane Sattler 's layer Ciliary body Ciliary processes Ciliary muscle Pars plicata Pars plana Iris Stroma Pupil Iris dilator muscle Iris sphincter muscle
Retina ( inner ) Layers Inner limiting membrane Nerve fiber layer Ganglion cell layer Inner plexiform layer Inner nuclear layer Outer plexiform layer Outer
nuclear layer External limiting membrane Layer of rods and cones Retinal pigment epithelium Cells Photoreceptor cells ( Cone cell , Rod cell ) → ( Horizontal
cell ) → Bipolar cell → ( Amacrine cell ) → Retina ganglion cell ( Midget cell , Parasol cell ,
</item 1>

<item 2>
Retina
The optic disc is at right , and the macula lutea is near the centre . The retina is stratified into distinct layers , each containing specific cell types or
cellular compartments that have metabolisms with different nutritional requirements . To satisfy these requirements , the ophthalmic artery bifurcates and
supplies the retina via two distinct vascular networks : the choroidal network , which supplies the choroid and the outer retina , and the retinal network ,
which supplies the retina 's inner layer . Circulatory mechanisms ( edit ) At first glance , one may think that the vertebrate retina is `` wired wrongly '' or
`` badly designed '' ; but in fact , the retina could not function if it were not inverted . The photoreceptor layer must be embedded in the retinal pigment
epithelium ( RPE ) , which performs at least seven vital functions , one of the most obvious being to supply oxygen and other necessary nutrients needed for the
photoreceptors to function . These nutrients include glucose , fatty acids , and retinal .
</item 2>

<item 3>
Uvea
uva , grape ) , also called the uveal layer , uveal coat , uveal tract , or vascular tunic , is the pigmented middle of the three concentric layers that make up
an eye . The name is possibly a reference to its reddish - blue or almost black colour , wrinkled appearance and grape - like size and shape when stripped
intact from a cadaveric eye . Its use as a technical term in anatomy and ophthalmology is relatively modern . Contents ( hide ) 1 Structure 1.1 Regions 2
Function 3 Pharmacology 4 Immunology 5 Clinical significance 6 References 7 External links Structure ( edit ) Regions ( edit ) The uvea is the vascular middle
layer of the eye . It is traditionally divided into three areas , from front to back , the : Iris Ciliary body Choroid Function ( edit ) The prime functions of
the uveal tract as a unit are : Nutrition and gas exchange : uveal vessels directly perfuse the ciliary body and iris , to support their metabolic needs , and
indirectly supply diffusible nutrients to the outer retina , sclera , and lens , which lack any intrinsic blood supply .
</item 3>

<item 4>
Retina
Inner nuclear layer -- contains the nuclei and surrounding cell bodies ( perikarya ) of the amacrine cells , bipolar cells , and horizontal cells . Outer
plexiform layer -- projections of rods and cones ending in the rod spherule and cone pedicle , respectively . These make synapses with dendrites of bipolar
cells . In the macular region , this is known as the Fiber layer of Henle . Outer nuclear layer -- cell bodies of rods and cones . External limiting membrane --
layer that separates the inner segment portions of the photoreceptors from their cell nucleus . Layer of rods and cones -- layer of rod cells and cone cells .
Retinal pigment epithelium -- single layer of cuboidal cells ( with extrusions not shown in diagram ) . This is closest to the choroid . These layers can be
grouped into 4 main processing stages : photoreception ; transmission to bipolar cells ; transmission to ganglion cells , which also contain photoreceptors ,
the photosensitive ganglion cells ; and transmission along the optic nerve . At each synaptic stage there are also laterally connecting horizontal and amacrine
cells .
</item 4>

<item 5>
Retina
The layers and anatomical correlation are as follows : OCT scan of the macular area of a retina at 800 nm , axial resolution 3 μm From innermost to outermost ,
the layers identifiable by OCT are as follows : # OCT Layer / Conventional Label Anatomical Correlate Reflectivity on OCT Specific anatomical boundaries ?
Additional references Posterior cortical vitreous Posterior cortical vitreous Hyper - reflective Yes Preretinal space In eyes where the vitreous has fully or
partially detached from the retina , this is the space created between the posterior cortical vitreous face and the internal limiting membrane of the retina .
Hypo - reflective Internal limiting membrane ( ILM ) Formed by Müller cell endfeet ( unclear if it can be observed on OCT ) Hyper - reflective No Nerve fiber
layer ( NFL ) Ganglion cell axons travelling towards the optic nerve Ganglion cell layer ( GCL ) Ganglion cell bodies ( and some displaced amacrine cells ) Hypo
- reflective 5 Inner plexiform layer ( IPL ) Synapses between bipolar , amacrine and ganglion cells Hyper - reflective 6 Inner nuclear layer ( INL ) a )
Horizontal ,
</item 5>

<item 6>
Uvea
( The cornea has no adjacent blood vessels and is oxygenated by direct gas exchange with the environment . ) Light absorption : the uvea improves the contrast
of the retinal image by reducing reflected light within the eye ( analogous to the black paint inside a camera ) , and also absorbs outside light transmitted
through the sclera , which is not fully opaque . In addition some uveal regions have special functions of great importance , including secretion of the aqueous
humour by the ciliary processes , control of accommodation ( focus ) by the ciliary body , and optimisation of retinal illumination by the iris 's control over
the pupil . Many of these functions are under the control of the autonomic nervous system . Pharmacology ( edit ) The pupil provides a visible example of the
neural feedback control in the body . This is subserved by a balance between the antagonistic sympathetic and parasympathetic divisions of the autonomic nervous
system . Informal pharmacological experiments have been performed on the pupil for centuries , since the pupil is readily visible , and its size can be readily
altered by applying drugs -- even crude plant extracts -- to the cornea .
</item 6>

<item 7>
Uvea
com ( hide ) Anatomy of the globe of the human eye Fibrous tunic ( outer ) Sclera Episcleral layer Schlemm 's canal Trabecular meshwork Cornea Limbus layers
Epithelium Bowman 's Stroma Dua 's layer Descemet 's Endothelium Uvea / vascular tunic ( middle ) Choroid Capillary lamina of choroid Bruch 's membrane Sattler
's layer Ciliary body Ciliary processes Ciliary muscle Pars plicata Pars plana Iris Stroma Pupil Iris dilator muscle Iris sphincter muscle Retina ( inner )
Layers Inner limiting membrane Nerve fiber layer Ganglion cell layer Inner plexiform layer Inner nuclear layer Outer plexiform layer Outer nuclear layer
External limiting membrane Layer of rods and cones Retinal pigment epithelium Cells Photoreceptor cells ( Cone cell , Rod cell ) → ( Horizontal cell ) → Bipolar
cell → ( Amacrine cell ) → Retina ganglion cell ( Midget cell , Parasol cell , Bistratified cell , Giant retina ganglion cells , Photosensitive ganglion cell )
→ Diencephalon : P cell , M cell , K cell ,
</item 7>

<item 8>
Uvea
M cell , K cell , Muller glia Other Macula Perifoveal area Parafoveal area Fovea Foveal avascular zone Foveola Optic disc Optic cup Ora serrata Anatomical
regions of the eye Anterior segment Adnexa Fibrous tunic Anterior chamber Aqueous humour Iris Posterior chamber Ciliary body Lens Capsule of lens Zonule of Zinn
Posterior segment Vitreous chamber Vitreous body Retina Choroid Other Keratocytes Ocular immune system Optical coherence tomography Eye care professional Eye
disease Refractive error Accommodation Physiological Optics Visual perception Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Uvea
Uvea - wikipedia Uvea Jump to : navigation , search For the Pacific island and chiefdom in Wallis and Futuna , see Wallis ( island ) and Uvea ( Wallis and
Futuna ) . For the Pacific commune and island in New Caledonia , see Ouvéa and Ouvéa Island . This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( February 2014 ) ( Learn how and when to
remove this template message ) Uvea Horizontal section of the eyeball . The constituents of the uvea follow : iris labeled at top , ciliary body labeled at
upper right , and choroid labeled at center right . ) Details Identifiers Latin tunica vasculosa bulbi MeSH D014602 TA A15. 2.03. 001 FMA 58103 Anatomical
terminology ( edit on Wikidata ) The uvea ( / ˈjuːvɪə / ) ( Lat .
</item 9>

<item 10>
Uvea
Pharmacological control over pupil size remains an important part of the treatment of some ocular diseases . Drugs can also reduce the metabolically active
process of secreting aqueous humour , which is important in treating both acute and chronic glaucoma . Immunology ( edit ) The normal uvea consists of immune
competent cells , particularly lymphocytes , and is prone to respond to inflammation by developing lymphocytic infiltrates . A rare disease called sympathetic
ophthalmia may represent ' cross-reaction ' between the uveal and retinal antigens ( i.e. , the body 's inability to distinguish between them , with resulting
misdirected inflammatory reactions ) . Clinical significance ( edit ) See uveitis , choroiditis , iritis , iridocyclitis , anterior uveitis , sympathetic
ophthalmia , and uveal melanoma . References ( edit ) Jump up ^ https://en.oxforddictionaries.com/definition/uvea Jump up ^ eye , human. `` Encyclopædia
Britannica from Encyclopædia Britannica 2006 Ultimate Reference Suite DVD 2009 External links ( edit ) MedlinePlus Encyclopedia 002337 Diagram at visionweb.
</item 10>
=== END ITEM id=23abe963ec738b9e ===

=== ITEM id=599bfad4155005b7 ===
Gold sentences and facts:
Sentence 0: `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper , spending five weeks at the top .
  Fact 0.0: 'Sacrifice' is a ballad performed by musician Elton John.
  Fact 0.1: The lyrics of 'Sacrifice' are by Bernie Taupin.
  Fact 0.2: The music of 'Sacrifice' is by Elton John.
  Fact 0.3: 'Sacrifice' appears on the 1989 album Sleeping with the Past.
  Fact 0.4: 'Sacrifice' was first released in October 1989.
  Fact 0.5: 'Sacrifice' was released in 1990.
  Fact 0.6: 'Sacrifice' was the second single from the album Sleeping with the Past.
  Fact 0.7: 'Sacrifice' achieved success, particularly in France and the UK.
  Fact 0.8: 'Sacrifice' became Elton John's first solo chart-topper in France and the UK.
  Fact 0.9: 'Sacrifice' spent five weeks at the top of the chart.

Retrieved items:
<item 1>
Sacrifice (song)
Sacrifice ( song ) - wikipedia Sacrifice ( song ) Jump to : navigation , search For other songs with the same title , see Sacrifice ( disambiguation ) § Songs .
`` Sacrifice '' Single by Elton John from the album Sleeping with the Past A-side `` Healing Hands '' ( second release ) B - side `` Love Is a Cannibal '' (
first release ) Released October 1989 ( first release ) May 1990 ( second release ) Format CD , vinyl record ( 7 '' and 12 `` ) , audio cassette Length 5 : 11
Label Rocket , MCA Songwriter ( s ) Elton John , Bernie Taupin Producer ( s ) Chris Thomas Elton John singles chronology `` Healing Hands '' ( 1989 ) ``
Sacrifice '' ( 1989 ) `` Club at the End of the Street '' ( 1990 ) `` Healing Hands '' ( 1989 ) `` Sacrifice '' ( 1989 ) `` Club at the End of the Street '' (
1990 ) `` Sacrifice '' is a ballad performed by musician Elton John . The lyrics are by Bernie Taupin and the music by John .
</item 1>

<item 2>
Sacrifice (song)
Official Charts Company . Official Charts Company . Retrieved 20 January 2016 . ^ Jump up to : `` Elton John singles , Billboard charts '' . allmusic .
Retrieved 9 September 2008 . Jump up ^ `` '' Sacrifice `` , Australian Singles Chart '' . australian - charts . Retrieved 13 December 2007 . Jump up ^ `` ''
Sacrifice `` , Dutch Mega Top 100 '' ( in Dutch ) . dutchcharts . Retrieved 11 April 2010 . Jump up ^ `` '' Sacrifice `` , French Singles Chart '' ( in French )
. lescharts . Retrieved 13 December 2007 . Jump up ^ German Singles Chart Charts-surfer.de Archived 8 January 2009 at the Wayback Machine . ( Retrieved 9
September 2008 ) Jump up ^ `` Irish Single Chart , database '' . irishcharts . Retrieved 9 September 2008 . Jump up ^ `` '' Sacrifice '' New Zealand Singles
Chart `` . charts.org.nz . Retrieved 11 April 2010 .
</item 2>

<item 3>
Sacrifice (song)
Awarded as The Best Song Retrieved from `` https://en.wikipedia.org/w/index.php?title=Sacrifice_(song)&oldid=833955469 '' Categories : 1989 singles 1990 singles
Disney songs Elton John songs European Hot 100 Singles number - one singles SNEP Top Singles number - one singles UK Singles Chart number - one singles Songs
with lyrics by Bernie Taupin Songs with music by Elton John Pop ballads Rock ballads Song recordings produced by Chris Thomas ( record producer ) 1989 songs MCA
Records singles The Rocket Record Company singles Music videos directed by Alek Keshishian Hidden categories : CS1 Dutch - language sources ( nl ) CS1 French -
language sources ( fr ) Webarchive template wayback links CS1 Norwegian - language sources ( no ) CS1 German - language sources ( de ) Articles with hAudio
microformats Use dmy dates from August 2010 Talk Contents About Wikipedia Dansk Español Français Italiano Nederlands Norsk nynorsk Polski Português Edit links
This page was last edited on 3 April 2018 , at 05 : 22 .
</item 3>

<item 4>
“Sacrifice” is a ballad by Elton John, with lyrics by Bernie Taupin, from the 1989 album *Sleeping with the Past*. Inspired by soul music, it portrays the end
of a marriage as “no sacrifice.” Initially released in 1989, it was reissued in 1990 as a double A-side with “Healing Hands,” becoming John’s first solo UK No.
1 and topping the chart for five weeks. It also reached No
</item 4>

<item 5>
Sacrifice (song)
Billboard Hot Adult Contemporary Tracks Chart ( 1990 ) Peak position Australian ARIA Singles Chart 7 Dutch Mega Top 100 Eurochart Hot 100 French SNEP Singles
Chart German Singles Chart 36 Irish Singles Chart New Zealand RIANZ Singles Chart 7 Norwegian Singles Chart Swiss Singles Chart 23 UK Singles Chart U.S.
Billboard Hot 100 18 Zimbabwe Singles Chart 6 Year - end charts ( edit ) Chart ( 1990 ) Rank Australian Singles Chart 47 Dutch Top 40 11 UK Singles Charts
Certifications and sales ( edit ) Country Certification Date Sales certified Physical sales France Silver 200,000 380,000 UK Platinum 1 September 1990 600,000
Personnel ( edit ) Elton John : vocals ,
</item 5>

<item 6>
Sacrifice (song)
The lyrics are by Bernie Taupin and the music by John . The song appears on the 1989 album Sleeping with the Past . It was first released in October 1989 , then
in 1990 , and was the second single from the album . It achieved success , particularly in France and the UK , where it became his first solo chart - topper ,
spending five weeks at the top . A country music cover of the song was performed by Don Henley and Vince Gill as part of the 2018 tribute album Restoration :
Reimagining the Songs of Elton John and Bernie Taupin . Contents ( hide ) 1 Overview 1.1 Background and writing 1.2 Music video 1.3 Chart performance 2 Track
listings 3 Charts and sales 3.1 Weekly charts 3.2 Year - end charts 3.3 Certifications and sales 4 Personnel 5 References 6 External links Overview ( edit )
Background and writing ( edit ) `` Sacrifice '' is inspired both by the Aretha Franklin song `` Do Right Woman , Do Right Man '' and described as a Percy Sledge
song by Elton John .
</item 6>

<item 7>
Sacrifice (song)
Jump up ^ `` UK certifications , database '' . Bpi . Retrieved 9 September 2008 . External links ( edit ) Lyrics of this song at MetroLyrics Elton John songs
1960s and 1970s `` Skyline Pigeon '' ( 1968 ) `` I 've Been Loving You '' ( 1968 ) `` Lady Samantha '' ( 1969 ) `` It 's Me That You Need '' ( 1969 ) `` Border
Song '' ( 1970 ) `` Rock and Roll Madonna '' ( 1970 ) `` From Denver to L.A. '' ( 1970 ) `` Take Me to the Pilot '' ( 1970 ) `` Your Song '' ( 1970 ) `` Friends
'' ( 1971 ) `` Levon '' ( 1971 ) `` Tiny Dancer '' ( 1972 ) `` Rocket Man '' ( 1972 ) `` Honky Cat '' ( 1972 ) `` I Think I 'm Going to Kill Myself '' ( 1972 )
`` Mona Lisas and Mad Hatters '' ( 1972 ) `` Crocodile Rock '' ( 1972 ) `` Elderberry Wine '' (
</item 7>

<item 8>
Sacrifice (song)
'' ( 1997 ) `` Something About the Way You Look Tonight '' ( 1997 ) `` Candle in the Wind 1997 '' ( 1997 ) `` Written in the Stars '' ( 1999 ) 2000s `` I Want
Love '' ( 2001 ) `` This Train Do n't Stop There Anymore '' ( 2002 ) `` Original Sin '' ( 2002 ) `` Are You Ready for Love '' ( 2003 ) `` Ghetto Gospel '' (
2005 ) `` Electricity '' ( 2005 ) `` The Bridge '' ( 2006 ) `` Joseph , Better You than Me '' ( 2008 ) `` Tiny Dancer ( Hold Me Closer ) '' ( 2009 ) Related
articles Albums discography Videography Book : Elton John Category : Elton John Ivor Novello Awards -- Best Song Musically and Lyrically 1960s She 's Leaving
Home ( 1968 ) 1970s Where Do You Go To ( My Lovely ) ? ( 1970 ) Something ( 1971 ) Do n't Let It Die ( 1972 ) Without You ( 1973 ) Daniel ( 1974 ) Streets
</item 8>

<item 9>
Sacrifice (song)
000 Personnel ( edit ) Elton John : vocals , Roland RD - 1000 digital piano Davey Johnstone : electric guitar Romeo Williams : bass Jonathan Moffett : drums Guy
Babylon : keyboards Fred Mandel : keyboards Preceded by `` Words '' by The Christians French SNEP number - one single 26 May 1990 -- 9 June 1990 ( 3 weeks )
Succeeded by `` Bo le lavabo ( WC Kiss ) '' by Lagaf ' Preceded by `` Vogue '' by Madonna Eurochart Hot 100 number - one single 16 June 1990 -- 28 July 1990 ( 7
weeks ) Succeeded by `` U Ca n't Touch This '' by MC Hammer Preceded by `` World In Motion '' by Englandneworder UK number - one single 17 June 1990 -- 21 July
1990 ( 5 weeks ) Succeeded by `` Turtle Power '' by Partners in Kryme References ( edit ) ^ Jump up to : `` Elton John - Full Official Chart History '' .
Official Charts Company . Official Charts Company . Retrieved 20 January 2016 .
</item 9>

<item 10>
Sacrifice (song)
The song is beloved by both John and Bernie Taupin and deemed a bookend of sorts to their first hit `` Your Song '' . Taupin has said he is surprised he wrote
it , and said in an interview , `` I think ' Sacrifice ' is one of the best songs we 've written . '' The song is not a typical love song , but rather a song
about a breakup of a marriage where the loss of the relationship is `` no sacrifice . '' Although the single was successful , Elton John did not tour Europe in
both 1989 -- 1990 to support the Sleeping with the Past album ( with the exception of one appearance in Knebworth Park on 30 June 1990 ) touring only in the
U.S. and Australia . The 1989 European tour was , instead , in support of the Reg Strikes Back album , with no material representing Sleeping with the Past . In
all shows that were done to support Sleeping with the Past in 1989 -- 1990 , Elton played this song solo on his Roland RD - 1000 Digital Piano .
</item 10>
=== END ITEM id=599bfad4155005b7 ===

=== ITEM id=d97addfdb29a721a ===
Gold sentences and facts:
Sentence 0: The park lies in the north - east of South Africa , in the eastern parts of Limpopo and Mpumalanga provinces . Phalaborwa , Limpopo is the only town in South Africa that borders the Kruger National Park . It is one of the largest national parks in the world , with an area of 19,485 square kilometres ( 7,523 sq mi ) . The park is approximately 360 kilometres ( 220 mi ) long , and has an average width of 65 kilometres ( 40 mi ) . At its widest point , the park is 90 kilometres ( 56 mi ) wide from east to west .
  Fact 0.0: The Kruger National Park lies in the north-east of South Africa.
  Fact 0.1: The Kruger National Park lies in the eastern parts of Limpopo and Mpumalanga provinces.
  Fact 0.2: Phalaborwa, Limpopo is the only town in South Africa that borders the Kruger National Park.
  Fact 0.3: The Kruger National Park is one of the largest national parks in the world.
  Fact 0.4: The Kruger National Park has an area of 19,485 square kilometres (7,523 sq mi).
  Fact 0.5: The Kruger National Park is approximately 360 kilometres (220 mi) long.
  Fact 0.6: The Kruger National Park has an average width of 65 kilometres (40 mi).
  Fact 0.7: At its widest point, the Kruger National Park is 90 kilometres (56 mi) wide from east to west.
Sentence 1: Kruger National Park is one of the largest game reserves in Africa . It covers an area of 19,485 square kilometres ( 7,523 sq mi ) in the provinces of Limpopo and Mpumalanga in northeastern South Africa , and extends 360 kilometres ( 220 mi ) from north to south and 65 kilometres ( 40 mi ) from east to west . The administrative headquarters are in Skukuza . Areas of the park were first protected by the government of the South African Republic in 1898 , and it became South Africa 's first national park in 1926 .
  Fact 1.0: Kruger National Park is one of the largest game reserves in Africa.
  Fact 1.1: Kruger National Park covers an area of 19,485 square kilometres (7,523 sq mi).
  Fact 1.2: Kruger National Park is in the provinces of Limpopo and Mpumalanga.
  Fact 1.3: Kruger National Park is in northeastern South Africa.
  Fact 1.4: Kruger National Park extends 360 kilometres (220 mi) from north to south.
  Fact 1.5: Kruger National Park extends 65 kilometres (40 mi) from east to west.
  Fact 1.6: The administrative headquarters of Kruger National Park are in Skukuza.
  Fact 1.7: Areas of Kruger National Park were first protected by the government of the South African Republic in 1898.
  Fact 1.8: Kruger National Park became South Africa's first national park in 1926.
Sentence 2: Kruger National reserve IUCN category II ( national park ) Burchell 's zebra in a Kruger landscape Location of the park ( red area ) in South Africa Location Limpopo and Mpumalanga provinces , South Africa Nearest city Nelspruit ( southern ) Phalaborwa ( central ) Coordinates 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Coordinates : 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Area 19,485 km ( 7,523 sq mi ) Established 31 May 1926 ( 1898 ) Visitors 1,336,981 ( in 2004 ) Governing body South African National Parks www.sanparks.org/parks/kruger/
  Fact 2.0: Kruger National Reserve is IUCN category II (national park).
  Fact 2.1: Kruger National Reserve is located in Limpopo and Mpumalanga provinces, South Africa.
  Fact 2.2: The nearest city to Kruger National Reserve is Nelspruit (southern).
  Fact 2.3: The nearest city to Kruger National Reserve is Phalaborwa (central).
  Fact 2.4: Kruger National Reserve has coordinates 24°0′41''S 31°29′7''E / 24.01139°S 31.48528°E.
  Fact 2.5: Kruger National Reserve has an area of 19,485 km (7,523 sq mi).
  Fact 2.6: The infobox lists Kruger National Reserve as established on 31 May 1926, with 1898 given in parentheses.
  Fact 2.7: Kruger National Reserve had 1,336,981 visitors in 2004.
  Fact 2.8: Kruger National Reserve is governed by South African National Parks.
  Fact 2.9: The website for Kruger National Reserve is www.sanparks.org/parks/kruger/.

Retrieved items:
<item 1>
Kruger National Park
It is one of the largest national parks in the world , with an area of 19,485 square kilometres ( 7,523 sq mi ) . The park is approximately 360 kilometres ( 220
mi ) long , and has an average width of 65 kilometres ( 40 mi ) . At its widest point , the park is 90 kilometres ( 56 mi ) wide from east to west . To the
north and south of the park two rivers , the Limpopo and the Crocodile respectively , act as its natural boundaries . To the east the Lebombo Mountains separate
it from Mozambique . Its western boundary runs parallel with this range , roughly 65 kilometres ( 40 mi ) distant . The park varies in altitude between 200
metres ( 660 ft ) in the east and 840 metres ( 2,760 ft ) in the south - west near Berg - en - Dal . The highest point in the park is here , a hill called
Khandzalive . Several rivers run through the park from west to east , including the Sabie , Olifants , Crocodile , Letaba , Luvuvhu and Limpopo Rivers .
</item 1>

<item 2>
Kruger National Park
The larger communities include Bosbokrand , Acornhoek , Hazyview , Hoedspruit , Komatipoort , Malelane , Marloth Park , Nelspruit and Phalaborwa . Accommodation
and facilities ( edit ) The Kruger National Park has 21 rest camps , as well as two private lodge concessions , and 15 designated private safari lodges . The
concessions are parcels of land operated by private companies in partnership with communities , who outsource the operation of private lodges . Camping in the
park has become popular with tourists and backpackers because it is much less expensive , and open to anyone , requiring no special permission to partake . Rest
camps ( edit ) Mopani 's view over Pioneer Dam . Orpen Rest Camp . A tent in Tamboti Tented Camp . Bateleur Bushveld Camp Berg - en - Dal Camp Biyamiti Bushveld
Camp Boulders Bush Camp Balule Camp , near Olifants Camp Crocodile Bridge Camp Letaba Camp Lower Sabie Camp Malelane Camp Maroela Camp ,
</item 2>

<item 3>
Kruger National Park
981 ( in 2004 ) Governing body South African National Parks www.sanparks.org/parks/kruger/ Kruger National Park is one of the largest game reserves in Africa .
It covers an area of 19,485 square kilometres ( 7,523 sq mi ) in the provinces of Limpopo and Mpumalanga in northeastern South Africa , and extends 360
kilometres ( 220 mi ) from north to south and 65 kilometres ( 40 mi ) from east to west . The administrative headquarters are in Skukuza . Areas of the park
were first protected by the government of the South African Republic in 1898 , and it became South Africa 's first national park in 1926 . To the west and south
of the Kruger National Park are the two South African provinces of Limpopo and Mpumalanga . In the north is Zimbabwe , and to the east is Mozambique . It is now
part of the Great Limpopo Transfrontier Park , a peace park that links Kruger National Park with the Gonarezhou National Park in Zimbabwe , and with the Limpopo
National Park in Mozambique .
</item 3>

<item 4>
Kruger National Park
fin24.com/Companies/TravelAndLeisure/Kruger-Park-closes-as-eateries-move-in-20131216 External links ( edit ) Find more aboutKruger National Parkat Wikipedia 's
sister projects Media from Wikimedia Commons Travel guide from Wikivoyage Map all coordinates using : OpenStreetMap Google Maps Download coordinates as : KML
GPX Kruger National Park official website Webcams in the Kruger National parks of South Africa Addo Elephant Agulhas ǀAi - ǀAis / Richtersveld Augrabies
Bontebok Camdeboo Garden Route Golden Gate Highlands Karoo Kgalagadi Kruger Mapungubwe Marakele Mokala Mountain Zebra Namaqua Table Mountain Tankwa Karoo
Tsitsikamma West Coast VIAF : 151887382 ISNI : 0000 0001 2186 0681 GND : 4098727 - 9 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Kruger_National_Park&oldid=837962811 '' Categories : IUCN Category II Protected areas established in 1926 Kruger
National Park National parks of South
</item 4>

<item 5>
Kruger National Park
1946 -- 1994 ( edit ) Plaque in the park . Now and then people do get killed , however , this is extremely rare . Stevenson - Hamilton was replaced by Colonel
J.A.B. Sandenbergh of the South African Air Force . During 1959 , work commenced to completely fence the park boundaries . Work started on the southern boundary
along the Crocodile River and in 1960 the western and northern boundaries were fenced , followed by the eastern boundary with Mozambique . The purpose of the
fence was to curb the spread of diseases , facilitate border patrolling and inhibit the movement of poachers . The Makuleke area in the northern part of the
park was forcibly taken from the Makuleke people by the government in 1969 and about 1500 of them were relocated to land to the south so that their original
tribal areas could be integrated into the greater Kruger National Park . 1994 -- present ( edit ) In 1996 the Makuleke tribe submitted a land claim for 19,842
hectares ( 198.42 km ) in the northern part of the Kruger National Park .
</item 5>

<item 6>
Kruger National Park
The land was given back to the Makuleke people , however , they chose not to resettle on the land but to engage with the private sector to invest in tourism ,
thus resulting in the building of several game lodges . In the late 1990s , the fences between the Kruger Park and Klaserie Game Reserve , Olifants Game Reserve
and Balule Game Reserve were dropped and incorporated into the Greater Kruger Park with 40 000 hectares added to the Reserve . In 2002 , Kruger National Park ,
Gonarezhou National Park in Zimbabwe , and Limpopo National Park in Mozambique were incorporated into a peace park , the Great Limpopo Transfrontier Park .
Location and geography ( edit ) Geography of the Kruger National Park Olifants River Luvuvhu and Limpopo rivers at Crookes Corner in Pafuri triangle The park
lies in the north - east of South Africa , in the eastern parts of Limpopo and Mpumalanga provinces . Phalaborwa , Limpopo is the only town in South Africa that
borders the Kruger National Park .
</item 6>

<item 7>
Kruger National Park
That motion , introduced together with another Volksraad member by the name of R.K. Loveday , and accepted for discussion in September 1895 by a majority of one
vote , resulted in the proclamation by Paul Kruger , president of the Transvaal Republic ( South African Republic ) , on 26 March 1898 , of a `` Government
Wildlife Park . '' This park would later be known as the Sabi Game Reserve . The park was initially created to control hunting and protect the diminished number
of animals in the park . James Stevenson - Hamilton became the first warden of the reserve in 1902 . The reserve was located in the southern one - third of the
modern park . Shingwedzi Reserve , named after the Shingwedzi River and now in northern Kruger National Park , was proclaimed in 1903 . During the following
decades all the native tribes were removed from the reserve and during the 1960s the last were removed at Makuleke in the Pafuri triangle . In 1926 , Sabie Game
Reserve , the adjacent Shingwedzi Game Reserve , and farms were combined to create Kruger National Park .
</item 7>

<item 8>
Kruger National Park
Kruger National Park - Wikipedia Kruger National Park Jump to : navigation , search Kruger National reserve IUCN category II ( national park ) Burchell 's zebra
in a Kruger landscape Location of the park ( red area ) in South Africa Location Limpopo and Mpumalanga provinces , South Africa Nearest city Nelspruit (
southern ) Phalaborwa ( central ) Coordinates 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Coordinates : 24 ° 0 ′
41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Area 19,485 km ( 7,523 sq mi ) Established 31 May 1926 ( 1898 ) Visitors
1,336,981 ( in 2004 ) Governing body South African National Parks www.sanparks.
</item 8>

<item 9>
Kruger National Park
3 Red bush - willow and mopane veld 4.1. 4 Shrub mopane veld 4.2 Birds 4.3 Mammals 4.4 Reptiles 4.5 Amphibians and fish 5 Anti-poaching measures 5.1 Poachers
5.2 Rhino horn 5.3 Ivory stocks 5.4 Other 6 Accommodation and facilities 6.1 Rest camps 6.2 Designated private lodges 6.3 Bushveld camps 6.4 Restaurants 7 Gates
to the Kruger Park 8 Wilderness trails 9 Gallery 10 See also 11 References 12 External links History ( edit ) Main article : History of the Kruger National Park
Pre-reserve ( -- 1898 ) ( edit ) The area that the park covers today was part of the last wild frontier in the eastern half of Transvaal before the Second Anglo
- Boer War . Paul Kruger , President of the Republic of South Africa at the time , proclaimed the area , which was inhabited by the Tsonga people , a sanctuary
for the protection of its wildlife .
</item 9>

<item 10>
Kruger National Park
and min . temperatures in ° F Precipitation totals in inches The climate of the Kruger National Park and Lowveld is subtropical . Summer days are humid and hot
. The rainy season is from September until May . The Kruger National Park website lists September and October as the driest periods , culminating in rains late
in October . The dry winter season is the ideal time to visit this region for various reasons . There is less chance of contracting malaria and the days are
milder . Viewing wildlife is more rewarding as the vegetation is more sparse and animals are drawn to the waterholes to drink every morning and evening . Flora
and Fauna ( edit ) Vegetation ( edit ) Plants life in the park consists of four main areas : Natural landscape for much of the Park Thorn trees and red bush -
willow veld ( edit ) This area lies between the western boundary and roughly the centre of the park south of the Olifants River .
</item 10>
=== END ITEM id=d97addfdb29a721a ===

=== ITEM id=2f0ac80d1534469c ===
Gold sentences and facts:
Sentence 0: A subcutaneous injection is administered as a bolus into the subcutis , the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut is the preferred abbreviation for patient safety .
  Fact 0.0: A subcutaneous injection is administered as a bolus into the subcutis.
  Fact 0.1: The subcutis is the layer of skin directly below the dermis and epidermis.
  Fact 0.2: The dermis and epidermis are collectively referred to as the cutis.
  Fact 0.3: Subcutaneous injections are highly effective in administering vaccines.
  Fact 0.4: Subcutaneous injections are highly effective in administering medications such as insulin, morphine, diacetylmorphine and goserelin.
  Fact 0.5: Subcutaneous injection of recreational drugs, as opposed to intravenous injection, is referred to as 'skin popping'.
  Fact 0.6: Subcutaneous administration may be abbreviated as SC, SQ, sub-cu, sub-Q, SubQ, or subcut.
  Fact 0.7: Subcut is the preferred abbreviation for patient safety.

Retrieved items:
<item 1>
Subcutaneous injection
25.2007. s112 . ( hide ) Routes of administration , dosage forms Oral Digestive tract ( enteral ) Solids Pill Tablet Capsule Pastille Time release technology
Osmotic delivery system ( OROS ) Liquids Decoction Elixir Electuary Emulsion Extended - release syrup Effervescent powder or tablet Herbal tea Hydrogel
Molecular encapsulation Powder Softgel Solution Suspension Syrup Syrup Concentrate for dilution and / or addition of carbonated water Tincture Buccal (
sublabial ) , sublingual Solids Orally disintegrating tablet ( ODT ) Film Lollipop Sublingual drops Lozenges Effervescent buccal tablet Chewing gum Liquids
Mouthwash Toothpaste Ointment Oral spray Respiratory tract Solids Smoking device Dry - powder inhaler ( DPI ) 0 0 Liquids Anaesthetic vaporizer Vaporizer
Nebulizer Metered - dose inhaler ( MDI ) Gas Oxygen mask and Nasal cannula Oxygen concentrator Anaesthetic machine Relative analgesia machine Ophthalmic ,
otologic ,
</item 1>

<item 2>
Subcutaneous injection
Subcutaneous injection - wikipedia Subcutaneous injection Jump to : navigation , search `` Subcut '' redirects here . For the song by Grand Mixer DXT featuring
Bill Laswell , see Aftermathematics . Play media 3D animation of a subcutaneous injection A subcutaneous injection is administered as a bolus into the subcutis
, the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in
administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of
recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut
is the preferred abbreviation for patient safety . Subcutaneous tissue has few blood vessels and so drugs injected here are for slow , sustained rates of
absorption . It is slower than intramuscular injections but still faster than intradermal injections .
</item 2>

<item 3>
Subcutaneous injection
The 3 / 8 '' and 5 / 8 '' needles are most commonly used . Usually , no more than 1 mL of solution is given , compared to intradermal injections , where no more
than 0.5 mL is usually given . Procedure ( edit ) Subcutaneous injections are inserted at 45 to 90 degree angles , depending on amount of subcutaneous tissue
present and length of needle - a shorter , 3 / 8 '' needle is usually inserted 90 degrees and a 5 / 8 '' needle is usually inserted at 45 degrees . Medication
is administered slowly , about 10 seconds / milliliter . Subcutaneous insulin injections in diabetes mellitus ( edit ) An insulin pump with a subcutaneous
injection site . A person with insulin - dependent ( either type 1 or type 2 ) diabetes mellitus typically injects insulin subcutaneously . The injection should
be given under the skin , into the fat layer .
</item 3>

<item 4>
Subcutaneous injections act slower than intramuscular but faster than intradermal injections. Sites include the upper arm, abdomen (fastest absorption; avoid a
2-inch area around the navel), thigh, upper back, and upper buttock (slowest absorption). Use a 25–31 gauge, 3/8–1-inch needle selected for the patient’s tissue
depth; 3/8- and 5/8-inch needles are common. Inject at 45–90
</item 4>

<item 5>
Subcutaneous injection
The injection should be given under the skin , into the fat layer . See also ( edit ) Intramuscular injection Intravenous injection Intradermal injection
References ( edit ) Jump up ^ `` subcutaneous injection '' at Dorland 's Medical Dictionary Jump up ^ `` ISMP 's List of Error - Prone Abbreviations , Symbols ,
and Dose Designations '' ( PDF ) . www.ismp.org. 2013 . Retrieved 13 May 2013 . ^ Jump up to : Taylor , C.R. , Lillis , C. , LeMone , P. , Lynn , P. ( 2011 )
Fundamentals of nursing : The art and science of nursing care . Philadelphia : Lippincott Williams & Wilkins , page 749 . Jump up ^ `` Insulin Administration ''
. Diabetes Care . 25 ( suppl 1 ) : s112 -- s115 . January 2002 . doi : 10.2337 / diacare. 25.2007. s112 . ( hide ) Routes of administration ,
</item 5>

<item 6>
Subcutaneous injection
otologic , nasal Nasal spray Ear drops Eye drops Ointment Hydrogel Nanosphere suspension Insufflation Mucoadhesive microdisc ( microsphere tablet ) Urogenital
Ointment Pessary ( vaginal suppository ) Vaginal ring Vaginal douche Intrauterine device ( IUD ) Extra-amniotic infusion Intravesical infusion Rectal ( enteral
) Ointment Suppository Enema Solution Hydrogel Murphy drip Nutrient enema Dermal Ointment Topical cream Topical gel Liniment Paste Film DMSO drug solution
Electrophoretic dermal delivery system Hydrogel Liposomes Transfersome vesicles Cream Lotion Lip balm Medicated shampoo Dermal patch Transdermal patch Contact (
rubbed into break in the skin ) Transdermal spray Jet injector Injection , infusion ( into tissue / blood ) Skin Intradermal Subcutaneous Transdermal implant
Organs Intracavernous Intravitreal Intra-articular injection Transscleral Central nervous system Intracerebral Intrathecal Epidural Circulatory ,
</item 6>

<item 7>
The text combines a brief guide to subcutaneous injections—uses, absorption, sites and needles—with broad catalogs of medication routes and drug classes, oral
conditions, digestive diseases, and procedures. Most material surveys digestive anatomy, development and function: gastrointestinal organs, bile and pancreatic
secretions, digestion, nutrient absorption, motility, blood supply, and the small and large intestines. It also covers liver, gallbladder, pancreas, mouth,
swallowing, and related disorders, diagnostics, and surgery
</item 7>

<item 8>
A subcutaneous injection delivers a medication or vaccine as a bolus into the fatty tissue beneath the skin’s outer layers. Because this tissue has few blood
vessels, absorption is slow and sustained—slower than intramuscular but faster than intradermal injection. Common uses include insulin, morphine, goserelin, and
vaccines. The route is abbreviated in several ways; “Subcut” is preferred for patient safety.
</item 8>

<item 9>
Subcutaneous injection
musculoskeletal Intravenous Intracardiac Intramuscular Intraosseous Intraperitoneal Nanocell injection Patient - Controlled Analgesia pump PIC line WikiProject
Pharmacy portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Subcutaneous_injection&oldid=827949994 '' Categories : Dosage forms Routes of
administration Injection ( medicine ) Talk Contents About Wikipedia Dansk Deutsch Español فارسی Français Italiano Nederlands Norsk Polski Português Slovenščina
Türkçe Українська Tiếng Việt 6 more Edit links This page was last edited on 27 February 2018 , at 18 : 16 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
Subcutaneous injection
It is slower than intramuscular injections but still faster than intradermal injections . Contents ( hide ) 1 Injection sites 2 Equipment 3 Procedure 4
Subcutaneous insulin injections in diabetes mellitus 5 See also 6 References Injection sites ( edit ) Subcutaneous injection sites Sites include , The outer
area of the upper arm . The abdomen , from the rib margin to the iliac crest and avoiding a 2 - inch circle around the navel . This has the fastest rate of
absorption among the sites . The front of the thigh , midway to the outer side , 4 inches below the top of the thigh to 4 inches above the knee . This has a
slower rate of absorption than the upper arm . The upper back . The upper area of the buttock , just behind the hip bone . This has the slowest rate of
absorption among the sites . Equipment ( edit ) A 25 to 31 gauge thick , 3 / 8 '' to 1 '' long needle can be used . The size is determined by the amount of
subcutaneous tissue present , which is based on patient build .
</item 10>
=== END ITEM id=2f0ac80d1534469c ===

=== ITEM id=8c4002ccb136fe44 ===
Gold sentences and facts:
Sentence 0: A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie , starring Jane Fonda and Lily Tomlin .
  Fact 0.0: A cover of a song by Stacey Markus is used at the end of episode nine, 'The Invitation', of Grace and Frankie.
  Fact 0.1: Grace and Frankie is a 2015 Netflix original series.
  Fact 0.2: Grace and Frankie stars Jane Fonda.
  Fact 0.3: Grace and Frankie stars Lily Tomlin.

Retrieved items:
<item 1>
Let My Love Open the Door
Chris Thomas Pete Townshend UK singles chronology `` Rough Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) `` Rough
Boys '' ( 1980 ) `` Let My Love Open the Door '' ( 1980 ) `` Keep on Working '' ( 1980 ) Pete Townshend US singles chronology `` Keep Me Turning '' ( 1977 ) (
with Ronnie Lane ) Keep Me Turning1977 `` Let My Love Open the Door '' ( 1980 ) Let My Love Open the Door1980 `` A Little Is Enough '' ( 1980 ) A Little Is
Enough 1980 `` Let My Love Open the Door '' is a song written and performed by Pete Townshend from his 1980 album Empty Glass . It reached the top ten in the
United States in that same year , reaching number nine . It reached number five in Canada .
</item 1>

<item 2>
Let My Love Open the Door
at 22 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
Let My Love Open the Door
It reached number five in Canada . Contents ( hide ) 1 Background 2 Chart performance 2.1 Weekly charts 2.2 Year - end charts 3 Personnel 4 Covers 5 List of
Films , TV Shows , and Other Media featuring the track 6 References 7 External links Background ( edit ) Although Townshend is a devotee to the religious guru
Meher Baba , he claimed in the liner notes of his Anthology CD that `` Jesus sings '' on the track . `` Let My Love Open the Door '' was released as the second
single from Empty Glass in Britain , where it was backed with the non-album tracks `` Classified '' and `` Greyhound Girl . '' The song was a minor British hit
, reaching # 46 . The song saw more success when it was released as the debut single from Empty Glass in America , where the song reached # 9 . It was Pete
Townshend 's only solo top 10 hit on the Billboard Hot 100 , but The Who 's song `` I Can See For Miles , '' which was written by Townshend , reached the same
position on the chart 13 years earlier .
</item 3>

<item 4>
Let My Love Open the Door
The group Rogue Wave also covered it in `` Cover me '' released in 2017 . List of films , TV shows , and other media featuring the track ( edit ) The song has
been used frequently in film , most notably the comedy genre , often as trailer music for movies such as Jerry Maguire and How Do You Know , and then as end
credit song for Mr. Deeds in 2002 ; Jersey Girl and Along Came Polly , both in 2004 . It was featured in the following films and TV series : Look Who 's Talking
( 1989 ) Grosse Pointe Blank as `` E. Cola Mix '' ( 1997 ) Mr.
</item 4>

<item 5>
Let My Love Open the Door
Let My Love Open the Door - wikipedia Let My Love Open the Door Jump to : navigation , search `` Let My Love Open the Door '' Single by Pete Townshend from the
album Empty Glass B - side `` Greyhound Girl '' Released June 4 , 1980 ( US ) Format 7 '' Recorded March 1980 Genre Art rock , new wave Length 2 : 44 Label Atco
Records Songwriter ( s ) Pete Townshend Producer ( s ) Pete Townshend ,
</item 5>

<item 6>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 6>

<item 7>
Let My Love Open the Door
Initially , Townshend 's manager despised the track due to it `` not sounding like Townshend , '' and wanted it to be removed from Empty Glass . However , upon
the song 's chart success , his manager called to apologize . Despite the song 's critical and commercial success , Pete Townshend did not consider it one of
his best songs . He told Rolling Stone in an interview that `` Let My Love Open the Door '' was `` just a ditty , '' also claiming that he preferred his minor
US hit `` A Little Is Enough '' from the same album . In 1996 , Townshend released a new version of the song , called `` the E. Cola mix '' , turning the song
into a ballad . This version appeared in different television shows and film soundtracks . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 1980 ) Peak
position Canada RPM 100 5 UK Singles Chart 46 U.S. Billboard Hot 100 9 U.S. Cash Box Top 100 11 Year - end charts ( edit ) Chart ( 1980 ) Rank Canada 35 U.S.
</item 7>

<item 8>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 8>

<item 9>
Let My Love Open the Door
A cover of the song by Stacey Markus is used at the end of episode nine ( `` The Invitation '' ) of the 2015 Netflix original series Grace and Frankie ,
starring Jane Fonda and Lily Tomlin . The song is featured in a Walmart Christmas holiday television commercial which started airing in early December 2017 .
Lethal Weapon ( TV series ) , season 2 , episode 9 `` Fools Rush In '' play the song performed by `` Rogue wave '' ( 2017 ) . References ( edit ) Jump up ^
Unterberger , Richie ( 2012 - 06 - 14 ) . `` Let My Love Open the Door '' . Rolling Stone . Retrieved 2016 - 10 - 16 . Jump up ^ `` Pete Townshend , ' Empty
Glass ' 100 Best Albums of the Eighties '' . Rolling Stone . 1989 - 11 - 16 . Retrieved 2016 - 10 - 16 . ^ Jump up to : Bac-lac.gc.ca ^ Jump up to : `` Let My
Love Open the Door '' . Songfacts.com .
</item 9>

<item 10>
Let My Love Open the Door
Psychoderelict O Parvardigar Related articles Songs Songs Written Albums Peter Meaden The Who Deep End Tommy ( film ) The Who 's Tommy Tommy ( EP ) Tommy (
soundtrack ) The Boy Who Heard Music Quadrophenia ( musical ) Quadrophenia ( film ) Quadrophenia ( soundtrack ) The Lifehouse Method Horse 's Neck Who I Am Eel
Pie Publishing Eel Pie Studios Double O Cliff Townshend Emma Townshend Simon Townshend The Wick Ashdown House , Oxfordshire The Boathouse , Twickenham Chapel
House , Twickenham Retrieved from `` https://en.wikipedia.org/w/index.php?title=Let_My_Love_Open_the_Door&oldid=838422442 '' Categories : 1980 singles Debut
singles Pete Townshend songs Songs written by Pete Townshend 1980 songs Atco Records singles Song recordings produced by Pete Townshend British new wave songs
Hidden categories : Webarchive template wayback links Music infoboxes with deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia
Čeština Español Euskara Edit links This page was last edited on 26 April 2018 ,
</item 10>
=== END ITEM id=8c4002ccb136fe44 ===

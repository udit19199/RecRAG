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

This file has 13 items.

=== ITEM id=e320f2b368927351 ===
Gold sentences and facts:
Sentence 0: Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
  Fact 0.0: Leycesteria is a genus of flowering plants.
  Fact 0.1: Leycesteria is in the honeysuckle family Caprifoliaceae.
  Fact 0.2: Leycesteria is native to temperate Asia in the Himalaya.
  Fact 0.3: Leycesteria is native to southwestern China.
Sentence 1: Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.
  Fact 1.0: Anigozanthos is a small genus of Australian plants.
  Fact 1.1: Anigozanthos is in the Bloodwort family Haemodoraceae.

Retrieved items:
<item 1>
Anigozanthos humilis
Anigozanthos humilis is a species of "Anigozanthos" in the family Haemodoraceae, known as common cat's paw or catspaw.

 This flowering perennial plant is endemic to Southwest Australia and widespread in its open forests.
</item 1>

<item 2>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 2>

<item 3>
Haemodoraceae
Haemodoraceae is a family of perennial herbaceous flowering plants with 14 genera and 102 known species.

 It is sometimes known as the "bloodwort family".

 Primarily a Southern Hemisphere family, they are found in South Africa, Australia and New Guinea, and in the Americas (from SE U.S.A. to tropical South
America).

 Perhaps the best known are the widely cultivated and unusual kangaroo paws from Australia, of the two closely related genera "Anigozanthos" and "Macropidia".
</item 3>

<item 4>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 4>

<item 5>
Anigozanthos
Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.

 The 11 species and several subspecies are commonly known as kangaroo paw and catspaw depending on the shape of their flowers.

 A further species, previously identified as "Anigozanthos fuliginosus" and commonly known as the black kangaroo paw, has been transferred to its own monotypic
genus and is now known as "Macropidia fuliginosa".
</item 5>

<item 6>
Common brown leafhopper
The common brown leafhopper, "Orosius orientalis" (Matsumura) (Hemiptera: Cicadellidae) is one of the most common species of Australian leafhoppers with a very
wide host range.

 It is an important vector of several viruses and phytoplasmas worldwide.

 In Australia, phytoplasmas vectored by "O. orientalis" cause a range of economically important diseases including legume little leaf, tomato big bud, lucerne
witches broom, potato purple top wilt, Australian lucerne and is a possible vector of Australian grapevine yellows.

 "O. orientalis" also transmits Tobacco yellow dwarf virus (TYDV genus Mastrevirus, family Geminiviridae) to beans, causing bean summer death disease and to
tobacco, causing tobacco yellow dwarf disease.
</item 6>

<item 7>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 7>

<item 8>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 8>

<item 9>
Anigozanthos preissii
Anigozanthos preissii, the Albany cat's paw, is a herbaceous plant species in the family Haemodoraceae, endemic to Western Australia.
</item 9>

<item 10>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 10>
=== END ITEM id=e320f2b368927351 ===

=== ITEM id=c9dea7ee4bde234f ===
Gold sentences and facts:
Sentence 0: Inner Secrets is the ninth studio album by Santana.
  Fact 0.0: Inner Secrets is a studio album by Santana.
  Fact 0.1: Inner Secrets is Santana's ninth studio album.
Sentence 1: Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana, which pioneered a fusion of rock and Latin American music.
  Fact 1.0: Carlos Santana was born on July 20, 1947.
  Fact 1.1: Carlos Santana is a Mexican musician.
  Fact 1.2: Carlos Santana is an American musician.
  Fact 1.3: Carlos Santana first became famous in the late 1960s and early 1970s.
  Fact 1.4: Carlos Santana first became famous with his band Santana.
  Fact 1.5: The band Santana pioneered a fusion of rock and Latin American music.

Retrieved items:
<item 1>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 1>

<item 2>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 2>

<item 3>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 3>

<item 4>
Chicago 13
Chicago 13 is the eleventh studio album by the American band Chicago, released in 1979.

 The follow-up to "Hot Streets", "Chicago 13" is often critically disfavored.

 This would be the band's final release that features lead guitarist Donnie Dacus, who had followed late, founding lead guitarist, Terry Kath.

 All band members would contribute to the songwriting (one of only two albums where this is the case, with the other being "Chicago VII").
</item 4>

<item 5>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 5>

<item 6>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 6>

<item 7>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 7>

<item 8>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 8>

<item 9>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 9>

<item 10>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 10>
=== END ITEM id=c9dea7ee4bde234f ===

=== ITEM id=3f1b0659ede256ff ===
Gold sentences and facts:
Sentence 0: William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story writer.
  Fact 0.0: William Somerset Maugham was born 25 January 1874.
  Fact 0.1: William Somerset Maugham died 16 December 1965.
  Fact 0.2: William Somerset Maugham is better known as W. Somerset Maugham.
  Fact 0.3: William Somerset Maugham was British.
  Fact 0.4: William Somerset Maugham was a playwright.
  Fact 0.5: William Somerset Maugham was a novelist.
  Fact 0.6: William Somerset Maugham was a short story writer.
  Fact 0.7: William Somerset Maugham's name includes the post-nominal CH.
Sentence 1:  He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
  Fact 1.0: W. Somerset Maugham was among the most popular writers of his era.
  Fact 1.1: W. Somerset Maugham was reputedly the highest-paid author during the 1930s.
Sentence 2: Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.
  Fact 2.0: Thomas Merton, O.C.S.O., was born January 31, 1915.
  Fact 2.1: Thomas Merton died December 10, 1968.
  Fact 2.2: Thomas Merton was an American Catholic writer.
  Fact 2.3: Thomas Merton was a theologian.
  Fact 2.4: Thomas Merton was a mystic.
Sentence 3:  In 1949, he was ordained to the priesthood and given the name Father Louis.
  Fact 3.0: In 1949, Thomas Merton was ordained to the priesthood.
  Fact 3.1: Thomas Merton was given the name Father Louis.

Retrieved items:
<item 1>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 1>

<item 2>
Trio (film)
Trio (also known as W. Somerset Maugham's Trio) is a 1950 British anthology film based on three short stories by W. Somerset Maugham: "The Verger", "Mr.
Know-All" and "Sanatorium".

 Ken Annakin directed "The Verger" and "Mr. Know-All", while Harold French was responsible for "Sanatorium".
</item 2>

<item 3>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 3>

<item 4>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 4>

<item 5>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 5>

<item 6>
Somerset Maugham Award
The Somerset Maugham Award is a British literary prize given each May by the Society of Authors.

 It is awarded to the best writer or writers of a book published in the past year who is under the age of 35.

 The prize was instituted in 1947 by William Somerset Maugham and thus bears his name.

 The award is to be spent on foreign travel.

 The total fund for each year is £12,000.
</item 6>

<item 7>
List of works about Thomas Merton
The following is a list of works about Thomas Merton, publications about Thomas Merton, the Trappist monk from Abbey of Gethsemani.

 The bibliography is organized into categories.

 A separate list of works by Thomas Merton is also available.
</item 7>

<item 8>
W. Somerset Maugham
William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story
writer.

 He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
</item 8>

<item 9>
Thomas Merton
Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.

 A Trappist monk of the Abbey of Gethsemani, Kentucky, he was a poet, social activist, and student of comparative religion.

 In 1949, he was ordained to the priesthood and given the name Father Louis.
</item 9>

<item 10>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 10>
=== END ITEM id=3f1b0659ede256ff ===

=== ITEM id=866d5d0199b41263 ===
Gold sentences and facts:
Sentence 0:  He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.
  Fact 0.0: The person described in this sentence was the brother of Alfred Jodl.
  Fact 0.1: Alfred Jodl was Chief of the Operations Staff of the OKW.
Sentence 1: Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
  Fact 1.0: Alfred Josef Ferdinand Jodl lived from 10 May 1890 to 16 October 1946.
  Fact 1.1: Alfred Jodl was a German general.
  Fact 1.2: Alfred Jodl was a war criminal during World War II.
  Fact 1.3: Alfred Jodl served as Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).

Retrieved items:
<item 1>
Hasso von Wedel (general)
Hasso von Wedel (20 November 1898 – 3 January 1961) was a German general who commanded the Wehrmacht Propaganda Troops during World War II.

 He was directly subordinate to the head of OKW Operations Staff ("Wehrmachtführungsstab, WFSt.")

, General Alfred Jodl.

 Wedel's Propaganda Department had control over the propaganda units and served to mediate between them and the Reich Propaganda Ministry of Joseph Goebbels.
</item 1>

<item 2>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 2>

<item 3>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 5>

<item 6>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 6>

<item 7>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 7>

<item 8>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 8>

<item 9>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 9>

<item 10>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 10>
=== END ITEM id=866d5d0199b41263 ===

=== ITEM id=da71c2cf3cd89175 ===
Gold sentences and facts:
Sentence 0: Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost figures of the French Avant-garde theatre.
  Fact 0.0: Eugène Ionesco was born Eugen Ionescu.
  Fact 0.1: Eugène Ionesco was born on 26 November 1909.
  Fact 0.2: Eugène Ionesco died on 28 March 1994.
  Fact 0.3: Eugène Ionesco was a Romanian-French playwright.
  Fact 0.4: Eugène Ionesco wrote mostly in French.
  Fact 0.5: Eugène Ionesco was one of the foremost figures of the French Avant-garde theatre.
Sentence 1: Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."
  Fact 1.0: Paul Mark Scott was born on 25 March 1920.
  Fact 1.1: Paul Mark Scott died on 1 March 1978.
  Fact 1.2: Paul Mark Scott was an English novelist.
  Fact 1.3: Paul Mark Scott was an English playwright.
  Fact 1.4: Paul Mark Scott was an English poet.
  Fact 1.5: Paul Mark Scott is best known for his monumental tetralogy "The Raj Quartet."

Retrieved items:
<item 1>
The Lesson
The Lesson (French: "La Leçon" ) is a one-act play by French-Romanian playwright Eugène Ionesco.

 It was first performed in 1951 in a production directed by Marcel Cuvelier (who also played the Professor).

 Since 1957 it has been in permanent showing at Paris' Théâtre de la Huchette, on an Ionesco double-bill with The Bald Soprano.

 The play is regarded as an important work in the "Theatre of the Absurd".
</item 1>

<item 2>
Paul Scott (novelist)
Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."

 His novel "Staying On" won the Booker Prize for 1977.
</item 2>

<item 3>
The Killer (play)
The Killer (French: Tueur sans gages , sometimes translated "The Killer without Reason" or "The Killer without Cause") is a play written by Eugène Ionesco in
1958.

 It is the first of Ionesco's Berenger plays, the others being "Rhinocéros" (1959), "Exit the King" (1962), and "A Stroll in the Air" (1963).
</item 3>

<item 4>
Eugène Ionesco
Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost
figures of the French Avant-garde theatre.

 Beyond ridiculing the most banal situations, Ionesco's plays depict the solitude and insignificance of human existence in a tangible way.
</item 4>

<item 5>
Victims of Duty
Victims of Duty (French: Victimes du Devoir ) is a one-act play written in 1953 by French-Romanian playwright Eugène Ionesco.

 An early work, it has not received the notoriety of his other works.

 The play is in the Theatre of the Absurd style, of which Ionesco was a pioneer.
</item 5>

<item 6>
The New Tenant
The New Tenant (French: "Le Nouveau Locataire" ) is a play written by Eugène Ionesco in 1955.

 The central image is common to many Ionesco plays: something accumulates on stage and overwhelms the characters.

 In this case its furniture.

 The main characters are a gentleman, a caretaker, and two movers.

 The caretaker talks as the gentleman, the "new tenant" of the title, directs the two movers who continuously bring in furniture.
</item 6>

<item 7>
Bill Scott (author)
William Neville "Bill" Scott OAM (4 October 1923 – 22 December 2005) was an Australian author, folklorist, songwriter, poet and a collector of bush ballads and
Australian folk history.

 He has published anthologies of Australian bush songs, including the best selling book "The Complete Book of Australian Folklore" published in 1976.

 He was awarded the Order of Australia in 1992 for his contributions to folklore, folk music and Australian literature.

 He was considered a living treasure and his anthologies of songs and his donated collections continue his legacy.
</item 7>

<item 8>
Exit the King
Exit the King (French: Le Roi se meurt ) is an absurdist drama by Eugène Ionesco that premiered in 1962.

 It is the third in Ionesco's "Berenger Cycle", preceded by "The Killer" (1958) and "Rhinocéros" (1959), and followed by "A Stroll in the Air" (1963).
</item 8>

<item 9>
Graph query row:
person: Paul Mark Scott
birth_date: 25 March 1920
death_date: 1 March 1978
</item 9>

<item 10>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 10>
=== END ITEM id=da71c2cf3cd89175 ===

=== ITEM id=50eca4066e56afe9 ===
Gold sentences and facts:
Sentence 0: William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story writer.
  Fact 0.0: William Somerset Maugham was born 25 January 1874.
  Fact 0.1: William Somerset Maugham died 16 December 1965.
  Fact 0.2: William Somerset Maugham is better known as W. Somerset Maugham.
  Fact 0.3: William Somerset Maugham was British.
  Fact 0.4: William Somerset Maugham was a playwright.
  Fact 0.5: William Somerset Maugham was a novelist.
  Fact 0.6: William Somerset Maugham was a short story writer.
  Fact 0.7: William Somerset Maugham's name includes the post-nominal CH.
Sentence 1:  He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
  Fact 1.0: W. Somerset Maugham was among the most popular writers of his era.
  Fact 1.1: W. Somerset Maugham was reputedly the highest-paid author during the 1930s.
Sentence 2: Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.
  Fact 2.0: Thomas Merton, O.C.S.O., was born January 31, 1915.
  Fact 2.1: Thomas Merton died December 10, 1968.
  Fact 2.2: Thomas Merton was an American Catholic writer.
  Fact 2.3: Thomas Merton was a theologian.
  Fact 2.4: Thomas Merton was a mystic.
Sentence 3:  In 1949, he was ordained to the priesthood and given the name Father Louis.
  Fact 3.0: In 1949, Thomas Merton was ordained to the priesthood.
  Fact 3.1: Thomas Merton was given the name Father Louis.

Retrieved items:
<item 1>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 1>

<item 2>
Trio (film)
Trio (also known as W. Somerset Maugham's Trio) is a 1950 British anthology film based on three short stories by W. Somerset Maugham: "The Verger", "Mr.
Know-All" and "Sanatorium".

 Ken Annakin directed "The Verger" and "Mr. Know-All", while Harold French was responsible for "Sanatorium".
</item 2>

<item 3>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 3>

<item 4>
W. Somerset Maugham
William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story
writer.

 He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
</item 4>

<item 5>
Somerset Maugham Award
The Somerset Maugham Award is a British literary prize given each May by the Society of Authors.

 It is awarded to the best writer or writers of a book published in the past year who is under the age of 35.

 The prize was instituted in 1947 by William Somerset Maugham and thus bears his name.

 The award is to be spent on foreign travel.

 The total fund for each year is £12,000.
</item 5>

<item 6>
Thomas Merton
Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.

 A Trappist monk of the Abbey of Gethsemani, Kentucky, he was a poet, social activist, and student of comparative religion.

 In 1949, he was ordained to the priesthood and given the name Father Louis.
</item 6>

<item 7>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 7>

<item 8>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 8>

<item 9>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 9>

<item 10>
List of works about Thomas Merton
The following is a list of works about Thomas Merton, publications about Thomas Merton, the Trappist monk from Abbey of Gethsemani.

 The bibliography is organized into categories.

 A separate list of works by Thomas Merton is also available.
</item 10>
=== END ITEM id=50eca4066e56afe9 ===

=== ITEM id=011592d637c315f6 ===
Gold sentences and facts:
Sentence 0: Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.
  Fact 0.0: Lowell J. Sherman was born on October 11, 1885.
  Fact 0.1: Lowell J. Sherman died on December 28, 1934.
  Fact 0.2: Lowell J. Sherman was an American actor.
  Fact 0.3: Lowell J. Sherman was an American film director.
Sentence 1: Jonathan Kaplan (born November 25, 1947) is an American film producer and director.
  Fact 1.0: Jonathan Kaplan was born on November 25, 1947.
  Fact 1.1: Jonathan Kaplan is an American film producer.
  Fact 1.2: Jonathan Kaplan is an American film director.

Retrieved items:
<item 1>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 1>

<item 2>
Jonathan Kaplan
Jonathan Kaplan (born November 25, 1947) is an American film producer and director.

 His film "The Accused" (1988) earned actress Jodie Foster her first Oscar for Best Actress and was nominated for the Golden Bear at the 39th Berlin
International Film Festival.

 Kaplan received five Emmy nominations for his roles directing and producing "ER".
</item 2>

<item 3>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 3>

<item 4>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 4>

<item 5>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 5>

<item 6>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 6>

<item 7>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 7>

<item 8>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 8>

<item 9>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 9>

<item 10>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 10>
=== END ITEM id=011592d637c315f6 ===

=== ITEM id=54a58a3333182c2e ===
Gold sentences and facts:
Sentence 0: Essence is a monthly magazine for African American women between the ages of 18 and 49.
  Fact 0.0: Essence is a monthly magazine.
  Fact 0.1: Essence is for African American women.
  Fact 0.2: Essence is aimed at women between the ages of 18 and 49.
Sentence 1: ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
  Fact 1.0: ALT for Damerne is a women's magazine.
  Fact 1.1: ALT for Damerne means 'All for the Ladies' in English.
  Fact 1.2: ALT for Damerne is written in the Danish language.
  Fact 1.3: ALT for Damerne is a weekly magazine.
  Fact 1.4: ALT for Damerne is published in Copenhagen, Denmark.

Retrieved items:
<item 1>
Alt for Damerne
ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
</item 1>

<item 2>
Essence Music Festival
The Essence Festival, known as "the party with a purpose", is an annual music festival which started in 1995 as a one-time event to celebrate the 25th
anniversary of "Essence", a magazine aimed primarily towards African-American women.

 It is the largest event celebrating African-American culture and music in the United States.

 Locally referred to as the Essence Fest, it has been held in New Orleans, Louisiana every year since 1994 except for 2006, when it was held in Houston, Texas
due to Hurricane Katrina's effect on New Orleans.

 It also held in Durban, KwaZulu-Natal in 2016.

 It features artists simultaneously performing on a main stage as well as four standing-room only superlounge stages.
</item 2>

<item 3>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 3>

<item 4>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 4>

<item 5>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 5>

<item 6>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 6>

<item 7>
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 7>

<item 8>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 8>

<item 9>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 9>

<item 10>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 10>
=== END ITEM id=54a58a3333182c2e ===

=== ITEM id=94aed753a0f9d879 ===
Gold sentences and facts:
Sentence 0: PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.
  Fact 0.0: PC Magazine is shortened as PCMag.
  Fact 0.1: PC Magazine is an American computer magazine.
  Fact 0.2: PC Magazine is published by Ziff Davis.
Sentence 1:  The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
  Fact 1.0: The magazine focuses on sculpture.
  Fact 1.1: The magazine focuses on sculpture collectors.
  Fact 1.2: The magazine focuses on sculpture enthusiasts.
  Fact 1.3: Painting is also frequently featured in the magazine.
  Fact 1.4: Poetry is also frequently featured in the magazine.

Retrieved items:
<item 1>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 1>

<item 2>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 2>

<item 3>
Mary Jo Foley
Mary Jo Foley is an American freelance technology writer, author, podcaster and news editor.

 She regularly writes news, previews, and reviews for Microsoft's strategy, products and technology.

 Foley has been covering news on Microsoft Windows, and previously on Unix-related technology, since 1983, for publications including "ZDNet", "eWeek",
"Baseline", "Redmond magazine" and "PC Magazine".
</item 3>

<item 4>
Computer Magazine
as well as "Computer Magazine" staff writers, and is currently managed and edited by the industry veteran Christopher Swearingin an MCSE and former CIO and
regarded author as well as contributor/reporter for "Computer Magazine" and other publications.
</item 4>

<item 5>
Windows: The Official Magazine
Windows: The Official Magazine is a technology magazine produced by Future plc in association with Microsoft, and published worldwide.

 The title describes itself as "A PC magazine for real life", and contains news, features, guides and reviews designed to show readers how to make better use of
their Windows PC.
</item 5>

<item 6>
Computer Press Association
Founded in 1983, the Computer Press Association (CPA) was established to promote excellence in the field of computer journalism.

 The association was composed of working editors, writers, producers, and freelancers who covered issues related to computers and technology.

 The CPA conducted the annual Computer Press Awards, which was the preeminent editorial awards of the computer and technology media.

 The CPA Awards honored outstanding examples in print, broadcast and electronic media.

 Awards were given for print publications, such as PC Magazine; online news media, such as Newsbytes News Network (both were multiple winners); individual
columns and features by well-known journalists such as Steven Levy (author of “”); broadcast awards such as “Best Radio Program”; as well as book awards in
categories such as Best Product Specific Book.

 CPA President Jeff Yablon (1994-1996) developed an updated code of ethics for technology journalists that was adopted by many major trade show groups, most
notably Bruno Blenheim.

 The Computer Press Association disbanded in 2000.
</item 6>

<item 7>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 7>

<item 8>
Lance Ulanoff
Lance Ulanoff is an American tech and social media commentator.

 He is a former Editor-in-Chief of "PCMag.com", "PC Magazine", and Mashable and SVP of Content for PCMag Digital Network, and is now an editor at Mashable.

 He spent nearly two decades in the computer technology publishing industry.

 Previously, he edited PCMag.com, the website for "PC Magazine".

 Ulanoff also writes an award-winning and popular column for the website.
</item 8>

<item 9>
Computer Magazine
Computer Magazine (and their website www.ComputerMagazine.com) is a popular magazine and online news site on computing and technology, offering current news and
reviews of popular and new business and consumer technologies, software, hardware, mobile computing, tablets, PCs, Macs, Windows, Linux, telecom, cellular,
wireless, data, cloud and science news on digital technologies and everything in the "tech-sphere and digi-verse", especially focused on information technology,
devices, software and services and related subjects, such as networking, servers, data centers and corporate data infrastructure technologies, and the Internet.

 Their online site, since 1997, is located at ComputerMagazine.com.

 "Computer Magazine" produces industry instructional and a popular ongoing webcast/podcast talk show and performs evaluations and reviews of IT industry
technology products, hardware, software and services with objective reporting widely respected as independent and objective, and trusted in the industry.
</item 9>

<item 10>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 10>
=== END ITEM id=94aed753a0f9d879 ===

=== ITEM id=741ef1201bc1e0a9 ===
Gold sentences and facts:
Sentence 0:  In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.
  Fact 0.0: In 2010, the person referred to as "he" began providing vocals in the pop rock band Ashland HIGH.
  Fact 0.1: In 2010, the person referred to as "he" began providing guitar in the pop rock band Ashland HIGH.
Sentence 1: Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.
  Fact 1.0: Jizzy Pearl was born James Wilkinson.
  Fact 1.1: Jizzy Pearl is an American vocalist.
  Fact 1.2: Jizzy Pearl was born on March 17, 1958.
  Fact 1.3: Jizzy Pearl was born in Chicago, Illinois.

Retrieved items:
<item 1>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 1>

<item 2>
Ron Cyrus
Ronald Ray "Ron" Cyrus (July 10, 1935 – February 28, 2006) was an American Democratic politician and public servant in Greenup County, Kentucky.

 He was the father of American country music singer/actor Billy Ray Cyrus and the paternal grandfather of Trace Cyrus, Miley Cyrus and Noah Cyrus.
</item 2>

<item 3>
Jizzy Pearl
Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.

 He's had notable performances in several heavy metal and hard rock groups.

 He first fronted the band Data Clan, which eventually became Love/Hate.

 Pearl has also sung for L.A. Guns, Ratt, Adler's Appetite, and other, lesser known acts.
</item 3>

<item 4>
Gold (Metro Station EP)
Gold is an EP by American pop rock band Metro Station released in 2014.

 It was the band's first release after Trace Cyrus' return to the group in 2014.
</item 4>

<item 5>
Trace Cyrus
Trace Dempsey Cyrus (born Neil Timothy Helson; February 24, 1989) is an American musician.

 The son of country singer Billy Ray Cyrus and brother of recording artist Miley Cyrus, he is the backing vocalist and bassist of the band Metro Station.

 He left the band in 2010 but returned in 2014.

 In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.

 He also owns the clothing company From Backseats to Bedrooms.
</item 5>

<item 6>
Brother Clyde (album)
Brother Clyde is the debut studio album from alternative rock group Brother Clyde.

 It is the first side-project from country music singer Billy Ray Cyrus.

 The album features duets with Cyrus' son Trace Cyrus as well as Dolly Parton, who Cyrus said "wanted to rock."

 It was released on August 10, 2010.

 Its first single, "Lately", which features rapper King Phaze, was issued to digital retailers on June 29, 2010, and a music video, which was directed by Cyrus,
was released for the song in July 2010.
</item 6>

<item 7>
Shrinking Violet (album)
Shrinking Violet is the sixth album by glam metal band L.A. Guns, first released on June 1, 1999, through Perris Records, and is the only L.A. Guns album with
singer Jizzy Pearl.

 The album was reissued, with bonus tracks and new artwork, on May 24, 2010, through Favored Nations.

 The album was produced by former Guns N' Roses guitarist Gilby Clarke who also provided additional guitars on the track "Dreamtime".
</item 7>

<item 8>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 8>

<item 9>
Metro Station (band)
Metro Station is an American pop rock band that was formed in Los Angeles by Trace Cyrus and Mason Musso.

 In late 2006, the band signed a recording contract with Columbia Records and RED Ink Records.

 The band is best known for the Top 10 "Billboard" hit single "Shake It" from the group's self-titled debut album.

 In 2010, tension between Cyrus and Musso caused the band to go on hiatus.

 In 2011 the band returned, however it was announced that Cyrus was no longer a part of the group and Musso had purchased the rights to the name.

 An EP entitled "Middle of the Night" was released in 2013, which was led by the single "Every Time I Touch You".

 In 2014 Cyrus and a new drummer, Spencer Steffan came to the band and a new single entitled "Love & War" was released.

 In 2015, the band released a second full-length album titled "Savior".
</item 9>

<item 10>
Noah Cyrus
Noah Lindsey Cyrus (born January 8, 2000) is an American actress and singer.

 She voiced the title role in the English version of the 2009 animated feature film "Ponyo".

 In 2016, she released her debut single "Make Me (Cry)", featuring vocals from Labrinth.

 Her debut album is set to be released in late 2017.

 She is the youngest daughter of Tish Cyrus and Billy Ray Cyrus and the younger sister of Miley Cyrus and Trace Cyrus.
</item 10>
=== END ITEM id=741ef1201bc1e0a9 ===

=== ITEM id=653f743003514850 ===
Gold sentences and facts:
Sentence 0: Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.
  Fact 0.0: Dangal's English title is "Wrestling competition".
  Fact 0.1: Dangal is a 2016 film.
  Fact 0.2: Dangal is an Indian film.
  Fact 0.3: Dangal is a Hindi-language film.
  Fact 0.4: Dangal is a biographical sports drama film.
  Fact 0.5: Dangal is loosely based on the story of Mahavir Singh Phogat.
  Fact 0.6: Mahavir Singh Phogat is an amateur wrestler.
  Fact 0.7: Mahavir Singh Phogat trains his daughters Geeta Phogat and Babita Kumari.
  Fact 0.8: Mahavir Singh Phogat trains his daughters to be world-class wrestlers.
Sentence 1: The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
  Fact 1.0: The Man from Snowy River II is a 1988 film.
  Fact 1.1: The Man from Snowy River II is an Australian drama film.
  Fact 1.2: The Man from Snowy River II is the sequel to the 1982 film "The Man from Snowy River".

Retrieved items:
<item 1>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 1>

<item 2>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 2>

<item 3>
The Man from Snowy River (poem)
"The Man from Snowy River" is a poem by Australian bush poet Banjo Paterson.

 It was first published in "The Bulletin", an Australian news magazine, on 26 April 1890, and was published by Angus & Robertson in October 1895, with other
poems by Paterson, in "The Man from Snowy River and Other Verses".
</item 3>

<item 4>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 4>

<item 5>
The Man from Snowy River (soundtrack)
"The Man from Snowy River is the original motion picture soundtrack from the 1982 film "The Man from Snowy River".
</item 5>

<item 6>
The Man from Snowy River II
The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
</item 6>

<item 7>
Graph query row:
film: Dangal
published_on: 2016
compared_film: The Man from Snowy River II
compared_published_on: 1988
connection: None
</item 7>

<item 8>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 8>

<item 9>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 9>

<item 10>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 10>
=== END ITEM id=653f743003514850 ===

=== ITEM id=d86329fc7352dfaa ===
Gold sentences and facts:
Sentence 0: Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.
  Fact 0.0: Alice Ann Munro's nee name is Laidlaw.
  Fact 0.1: Alice Ann Munro was born on 10 July 1931.
  Fact 0.2: Alice Ann Munro is a Canadian short story writer.
  Fact 0.3: Alice Ann Munro is a Nobel Prize winner.
Sentence 1: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.0: Lillian Florence Hellman was born on June 20, 1905.
  Fact 1.1: Lillian Florence Hellman died on June 30, 1984.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.
  Fact 1.3: Lillian Florence Hellman was an American screenwriter.
  Fact 1.4: Lillian Florence Hellman is known for her success as a playwright on Broadway.
  Fact 1.5: Lillian Florence Hellman is known for her left-wing sympathies.
  Fact 1.6: Lillian Florence Hellman is known for her political activism.

Retrieved items:
<item 1>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 1>

<item 2>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 2>

<item 3>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 3>

<item 4>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 4>

<item 5>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 5>

<item 6>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 6>

<item 7>
Graph query row:
person: Alice Munro
occupation: short story writer
relationship: None
related_work: None
</item 7>

<item 8>
Anne Perry
Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk
series.

 At the age of fifteen she was convicted of participating in the murder of her friend's mother, in 1954.

 She changed her name after serving her five-year sentence.
</item 8>

<item 9>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 9>

<item 10>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 10>
=== END ITEM id=d86329fc7352dfaa ===

=== ITEM id=13836c93ff576052 ===
Gold sentences and facts:
Sentence 0:  His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files" (1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
  Fact 0.0: The subject of this sentence portrayed Bret Maverick in the series Maverick from 1957 to 1962.
  Fact 0.1: The subject of this sentence portrayed Bret Maverick in the series Bret Maverick from 1981 to 1982.
  Fact 0.2: The subject of this sentence portrayed Jim Rockford in the series The Rockford Files from 1974 to 1980.
  Fact 0.3: The subject of this sentence portrayed Jim Egan in the series 8 Simple Rules from 2003 to 2005.
Sentence 1: The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.
  Fact 1.0: The Notebook is a 2004 film.
  Fact 1.1: The Notebook is an American romantic drama film.
  Fact 1.2: The Notebook was directed by Nick Cassavetes.
  Fact 1.3: The Notebook is based on a 1996 novel.
  Fact 1.4: The 1996 novel has the same name as the film The Notebook.
  Fact 1.5: The 1996 novel was written by Nicholas Sparks.

Retrieved items:
<item 1>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 1>

<item 2>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 2>

<item 3>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 3>

<item 4>
Mister Buddwing
Mister Buddwing is a 1966 American film drama directed by Delbert Mann and starring James Garner.
</item 4>

<item 5>
Bride and Prejudice
Bride and Prejudice is a 2004 romantic drama film directed by Gurinder Chadha.

 The screenplay by Chadha and Paul Mayeda Berges is a Bollywood-style adaptation of "Pride and Prejudice" by Jane Austen.

 It was filmed primarily in English, with some Hindi and Punjabi dialogue.

 The film released in the United States on 11 February 2005 and was well received by critics.
</item 5>

<item 6>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 6>

<item 7>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 7>

<item 8>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 8>

<item 9>
Milan (2004 film)
Milan is a 2004 romantic drama film released under Star Cinema, ABS-CBN Film Productions, Inc. in the Philippines.

 It stars Claudine Barretto and Piolo Pascual.

 It is a love story written by Raymond Lee, and under Olivia Lamasan's direction.

 The movie made more than P100 million in the box office.
</item 9>

<item 10>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 10>
=== END ITEM id=13836c93ff576052 ===

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

This file has 12 items.

=== ITEM id=69fef15c552ebb9b ===
Gold sentences and facts:
Sentence 0: The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.
  Fact 0.0: The Battle of Okinawa is also known as 沖縄戦 (Okinawa-sen in Hepburn romanization) and as 'Uchinaa ikusa' in Okinawan.
  Fact 0.1: The Battle of Okinawa was codenamed Operation Iceberg.
  Fact 0.2: The Battle of Okinawa was a major battle of the Pacific War.
  Fact 0.3: The Battle of Okinawa was fought on the island of Okinawa.
  Fact 0.4: The Battle of Okinawa was fought by United States Marine and Army forces against the Imperial Japanese Army.
  Fact 0.5: The Battle of Okinawa included the largest amphibious assault in the Pacific War during World War II.
  Fact 0.6: The largest amphibious assault in the Pacific War was the April 1, 1945 invasion of Okinawa itself.
Sentence 1: Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.
  Fact 1.0: Operation Diadem was also referred to as the Fourth Battle of Monte Cassino.
  Fact 1.1: In Canada, Operation Diadem was called the Battle of the Liri Valley.
  Fact 1.2: Operation Diadem was an offensive operation undertaken by the Allies of World War II.
  Fact 1.3: The Allies in Operation Diadem included the U.S. Fifth Army.
  Fact 1.4: The Allies in Operation Diadem included the British Eighth Army.
  Fact 1.5: Operation Diadem took place in May 1944.
  Fact 1.6: Operation Diadem was part of the Italian Campaign of World War II.

Retrieved items:
<item 1>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 1>

<item 2>
Operation Diadem
Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by
the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.

 "Diadem" was supported by air attacks called Operation Strangle.

 The opposing force was the German 10th Army.
</item 2>

<item 3>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 3>

<item 4>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 4>

<item 5>
Operation Kikusui
Operation Kikusui (菊水作戦 , Kikusui sakusen ) was a series of suicidal air attacks by Imperial Japanese forces during the Battle of Okinawa against Allied fleets
in the waters around Okinawa, as part of Operation Ten-Go.

 The name of the operation, "Kikusui", comes from the hata-jirushi of the samurai Kusunoki Masashige.
</item 5>

<item 6>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 6>

<item 7>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 7>

<item 8>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 8>

<item 9>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 9>

<item 10>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 10>
=== END ITEM id=69fef15c552ebb9b ===

=== ITEM id=7dfeb976a7907d69 ===
Gold sentences and facts:
Sentence 0: Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in 1995.
  Fact 0.0: Haunted is the second album by American singer-songwriter Poe.
  Fact 0.1: Haunted was released in 2000.
  Fact 0.2: Haunted was released after a five-year hiatus from Poe's debut album 'Hello', which came out in 1995.
Sentence 1:  The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
  Fact 1.0: Haunted was self-produced.
  Fact 1.1: Haunted was created as a tribute to Poe's father.
  Fact 1.2: Haunted was created as a counterpart to the novel 'House of Leaves' by Poe's brother Mark Z. Danielewski.
Sentence 2:  A bestseller, it has been translated into a number of languages, and is followed by a companion piece, "The Whalestoe Letters".
  Fact 2.0: Haunted is a bestseller.
  Fact 2.1: Haunted has been translated into a number of languages.
  Fact 2.2: Haunted is followed by a companion piece, 'The Whalestoe Letters'.

Retrieved items:
<item 1>
Uncle Wiggily in Connecticut
"Uncle Wiggily in Connecticut" is a short story by J. D. Salinger, which appears in his collection "Nine Stories".

 It was originally published in the March 20, 1948 issue of "The New Yorker".
</item 1>

<item 2>
The Padlock
The Padlock is a two-act 'afterpiece' opera by Charles Dibdin.

 The text was by Isaac Bickerstaffe.

 It debuted in 1768 at the Drury Lane Theatre in London as a companion piece to "The Earl of Warwick".

 It partnered other plays before a run of six performances in tandem with "The Fatal Discovery" by John Home.

 "The Padlock" was a success, largely due to Dibdin's portrayal of Mungo, a blackface caricature of a black servant from the West Indies.

 The company took the production to the United States the next year, where a portrayal by Lewis Hallam, Jr. as Mungo met with even greater accolades.

 The libretto was first published in London in 1768 (?)

 and in Dublin in 1775.

 The play remained in regular circulation in the U.S. as late as 1843.

 It was revived by the Old Vic Company in London and on tour in the UK in 1979 in a new orchestration by Don Fraser and played in a double-bill with Garrick's
"Miss In Her Teens".
</item 2>

<item 3>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 3>

<item 4>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 4>

<item 5>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 5>

<item 6>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 6>

<item 7>
A Predicament
"A Predicament" is a humorous short story by Edgar Allan Poe, usually combined with its companion piece "How to Write a Blackwood Article."

 It was originally titled "The Scythe of Time".

 The paired stories parody the Gothic sensation tale, popular in England and America since the early 19th century.
</item 7>

<item 8>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 8>

<item 9>
Haunted (Poe album)
Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in
1995.

 The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
</item 9>

<item 10>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 10>
=== END ITEM id=7dfeb976a7907d69 ===

=== ITEM id=b39a41635983b86f ===
Gold sentences and facts:
Sentence 0: Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.
  Fact 0.0: Pagliacci is an Italian opera.
  Fact 0.1: Pagliacci has a prologue and two acts.
  Fact 0.2: Pagliacci has music and libretto by Ruggero Leoncavallo.
  Fact 0.3: The literal translation of Pagliacci is 'Clowns'.
Sentence 1: Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.
  Fact 1.0: Suor Angelica (Sister Angelica) is an opera.
  Fact 1.1: Suor Angelica is in one act.
  Fact 1.2: Suor Angelica is by Giacomo Puccini.
  Fact 1.3: Suor Angelica has an original Italian libretto by Giovacchino Forzano.

Retrieved items:
<item 1>
Jane Henschel
Jane Henschel (born 2 March 1952) is an American operatic mezzo soprano.

 Henschel, who was born in Wisconsin, studied at the University of Southern California, and then pursued further studies in Germany, where she has made her
home.

 Her numerous opera appearances include Baba the Turk in Igor Stravinsky's "The Rake’s Progress" with Glyndebourne Festival Opera, the Saito Kinen Festival
Matsumoto, and the Salzburg festival; Brangäne in Richard Wagner’s "Die Walküre" with Paris Opéra and the Los Angeles Opera; the Principessa in Giacomo
Puccini’s "Suor Angelica" with conductor Riccardo Chailly and the Royal Concertgebouw Orchestra; Blanche de la Force in Francis Poulenc’s "Dialogues des
Carmélites" in Amsterdam; Kostelnicka Buryjovka in Leoš Janáček’s "Jenůfa" under Seiji Ozawa in Japan; and the Kabanicka in Janáček’s "Katya Kabanova" at the
Salzburg Festival among others.
</item 1>

<item 2>
Norine Burgess
Norine Burgess is a Canadian singer.

 She is a graduate of the University of Calgary and the University of Toronto’s Opera School, mezzo-soprano.

 She received additional training as a member of the Canadian Opera Company (COC) Ensemble where she appeared in Electra, Suor Angelica, Lulu and Der
Rosenkavalier.

 Ms. Burgess also performed Le Nozze di Figaro(Cherubino), Ariadne auf Naxos (Dryad), La Traviata(Flora) and Die Zauberflöte (Second Lady) with the COC.

 Additional operatic successes include Le Nozze di Figaro (Cherubino) and Carmen (Mercedes) with the Vancouver Opera, Fenena in Nabucco with Manitoba Opera,
Albert Herring (Nancy) with the Calgary Opera and many appearances with the Edmonton Opera.
</item 2>

<item 3>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 3>

<item 4>
Anna Nechaeva
Anna Nechaeva is Russian soprano singer who was born in Saratov and used attend its Conservatory in 1996.

 Later she was offered to perform the role of "Tatiana" in "Eugene Onegin" at the Saint Petersburg Conservatory and by 2003 became a soloist at the Saint
Petersburg Opera where she continued her original role as well as other title roles in Giacomo Puccini's "Gianni Schicchi", Madama Butterfly, and "Suor
Angelica".

 From 2008 to 2011 she became a soloist at the Mikhaylovsky Theatre where she performed roles of "Nedda" in "Pagliacci" and "Rachel" in "The Jews" as well as
the title roles of "Rusalka" and of course "Tatiana" in "Eugene Onegin".

 In 2012, she made her first public appearance with Bolshoi Theatre where she sang in "The Enchantress" portraying "Nastasya" becoming soloist there the same
year.
</item 4>

<item 5>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 5>

<item 6>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 6>

<item 7>
Marie-Josée Lord
Marie-Josée Lord is a Haitian-born Canadian soprano.

 Lord was adopted from Haiti at the age of six and grew up in Lévis.

 She made her professional debut as Liù in Turandot in 2003 at the Opéra de Québec.

 She was particularly noted for her Suor Angelica in 2006.

 Her debut album of arias was released in 2011, becoming one of the Canadian label Atma's best-selling recordings.
</item 7>

<item 8>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 8>

<item 9>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 9>

<item 10>
Giovacchino Forzano
Giovacchino Forzano (] ; 19 November 1884 – 28 October 1970) was an Italian playwright, librettist, stage director, and film director.

 A resourceful writer, he authored numerous popular plays and produced opera librettos for most of the major Italian composers of the early twentieth century,
including the librettos for Giacomo Puccini's "Suor Angelica" and "Gianni Schicchi".
</item 10>
=== END ITEM id=b39a41635983b86f ===

=== ITEM id=77b73a3411dd9120 ===
Gold sentences and facts:
Sentence 0: Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and songwriter.
  Fact 0.0: Toshimitsu Deyama was born October 10, 1965 in Tateyama, Chiba.
  Fact 0.1: Toshimitsu Deyama's Japanese name is 出山利三.
  Fact 0.2: Toshimitsu Deyama's name is also written as Deyama Toshimitsu.
  Fact 0.3: Toshimitsu Deyama is known exclusively by his stage name Toshi.
  Fact 0.4: Toshimitsu Deyama is a Japanese singer and songwriter.
Sentence 1:  He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.
  Fact 1.0: Toshimitsu Deyama is best known as the lead vocalist of the heavy metal band X Japan.
  Fact 1.1: Toshimitsu Deyama is a co-founder of X Japan.
  Fact 1.2: X Japan disbanded in 1997.
Sentence 2: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 2.0: Warrel Dane was born March 7, 1969.
  Fact 2.1: Warrel Dane is the lead singer of the American power metal band Sanctuary.

Retrieved items:
<item 1>
Dreaming Neon Black
Dreaming Neon Black is the third full-length album released by Seattle progressive metal band Nevermore, and was released through Century Media in 1999.

 Unlike its predecessor, "The Politics of Ecstasy", this album is very emotional and contains many slower, ballad-type songs.

 It is also notable that "Dreaming Neon Black" is a concept album of sorts.

 According to Nevermore's lead singer, Warrel Dane, "it's a very simple story about a man who slowly goes insane after losing a woman that he was very close to.

 Progressive levels of insanity are expressed in the songs, he goes through phases of denial and self-blame, blaming God, then denouncing God.

 The ending is a little...tragic, a little depressing.

 Shakesperian.

 Everybody dies, it's all happy."

, which may be based on an event in the life of Dane.

 Supposedly, his old girlfriend left him when she joined a religious cult and was never heard from again, and he began having nightmares of her crying out to
him as she drowned.

 This has been confirmed by Warrel himself in an older interview.
</item 1>

<item 2>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 2>

<item 3>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 3>

<item 4>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 4>

<item 5>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 5>

<item 6>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 6>

<item 7>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 7>

<item 8>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Into the Mirror Live
Into The Mirror Live / Black Reflections is live promotional EP of the band Sanctuary recorded on May 12, 1990 at The Country Club in Reseda, California, during
their "Into the Mirror Black" tour.

 Approximately 1000 CD copies of this were made and less than 500 were ever distributed to radio stations and record stores.

 Because of contractual problems with Epic Records, the full version was never released for public consumption.

 Epic Records destroyed 500+ copies of this CD.

 Warrel Dane has stated on several occasions that he has the master tapes and copyrights of the full show and intended to have it finally released.

 To date that has not happened.

 This CD has been heavily pirated over the years as a 2-on-1 CD with the Satan's Host "Metal From Hell" album on the notorious Reborn Classics record label.
</item 10>
=== END ITEM id=77b73a3411dd9120 ===

=== ITEM id=2d87d2902e4c7c1c ===
Gold sentences and facts:
Sentence 0: The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver Division, the Shenango Division, and the Conneaut Division.
  Fact 0.0: The Beaver and Erie Canal was also known as the Erie Extension Canal.
  Fact 0.1: The Beaver and Erie Canal was part of the Pennsylvania Canal system.
  Fact 0.2: The Beaver and Erie Canal consisted of three sections.
  Fact 0.3: The three sections of the Beaver and Erie Canal were the Beaver Division, the Shenango Division, and the Conneaut Division.
Sentence 1: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.
  Fact 1.0: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp.
  Fact 1.1: The Dismal Swamp Canal is located in Virginia and North Carolina.
  Fact 1.2: Virginia and North Carolina are in the United States.

Retrieved items:
<item 1>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 1>

<item 2>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 2>

<item 3>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 3>

<item 4>
Great Dismal Swamp National Wildlife Refuge
The Great Dismal Swamp National Wildlife Refuge was created in 1974 to help protect and preserve a portion of the Great Dismal Swamp, a marshy region on the
Coastal Plain of southeastern Virginia and northeastern North Carolina between Norfolk, Virginia, and Elizabeth City, North Carolina in the United States.

 It is located in parts of the independent cities of Chesapeake and Suffolk in Virginia, and the counties of Camden, Gates, and Pasquotank in North Carolina.
</item 4>

<item 5>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 5>

<item 6>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 6>

<item 7>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 7>

<item 8>
Dismal Swamp State Park
Dismal Swamp State Park is a North Carolina state park in Camden County, North Carolina in the United States.

 The park was created as a state natural area in 1974 with the help of The Nature Conservancy, and on July 28, 2007 the NC General Assembly re-designated it as
a state park.

 It opened to the public in 2008.

 This marked the first time that public access to Great Dismal Swamp was made possible in North Carolina.

 The park covers 14432 acre of protected land on the North Carolina/Virginia border.

 Park offices are three miles (5 km) south of the border on U.S. Route 17 near South Mills.

 Features of the park include the canal which is used regularly by boaters using the Intracoastal Waterway and several miles of hiking and biking trails.
</item 8>

<item 9>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 9>

<item 10>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 10>
=== END ITEM id=2d87d2902e4c7c1c ===

=== ITEM id=a4694c599896c9e5 ===
Gold sentences and facts:
Sentence 0: Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
  Fact 0.0: Stephen Lawrence 'Steve' Winwood was born 12 May 1948.
  Fact 0.1: Steve Winwood is an English musician.
  Fact 0.2: Steve Winwood's genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
Sentence 1: Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.
  Fact 1.0: Per 'Pelle' Almqvist was born 29 May 1978.
  Fact 1.1: Per 'Pelle' Almqvist is also known as Howlin' Pelle Almqvist.
  Fact 1.2: Per 'Pelle' Almqvist is a Swedish singer and songwriter.

Retrieved items:
<item 1>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 1>

<item 2>
Revolutions – The Very Best of Steve Winwood
Revolutions – The Very Best of Steve Winwood is the sixth compilation album by Steve Winwood.

 The album includes music from Winwood's solo career, as well as groups with which he has performed, including The Spencer Davis Group, Traffic, and Blind
Faith.

 The CD was released as a box set and a single disc.

 As of October 2014, the box set is out of print (but can be purchased digitally), while the single disc is still available.

 The songs "The Finer Things" and "Roll With It" are exclusive to the single disc version and cannot be found on the box set.
</item 2>

<item 3>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 3>

<item 4>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 4>

<item 5>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 5>

<item 6>
Pelle Almqvist
Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.

 He is the lead singer of garage rock band The Hives.
</item 6>

<item 7>
Vince Guaraldi
Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions
and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of
Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
</item 7>

<item 8>
Paul Seelig
Paul Seelig (5 October 1900 – June 1931) was a Swedish actor.

 He appeared in about 15 roles in films 1913 and 1931.

 His film debut was in Mauritz Stiller's film "En pojke i livets strid" in 1913, when he was thirteen years old.
</item 8>

<item 9>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 9>

<item 10>
Steve Winwood (album)
Steve Winwood is the debut solo studio album by blue-eyed soulster Steve Winwood.

 It was released three years after the break-up of Traffic.

 Though the album sold moderately well in the USA, it was a commercial disappointment compared to Traffic's recent albums, peaking at number 22 on the
"Billboard" 200 album chart.

 In Winwood's home country the situation was reversed; while Traffic's recent albums had only been moderately successful in the UK, "Steve Winwood" reached
number 12 on The Official Charts.

 Island Records launched two singles from the album, "Hold On" and "Time Is Running Out", both of which failed to make the charts.
</item 10>
=== END ITEM id=a4694c599896c9e5 ===

=== ITEM id=1d1e5030aec462db ===
Gold sentences and facts:
Sentence 0: The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.
  Fact 0.0: The Celluloid Closet is a 1995 American documentary film.
  Fact 0.1: The Celluloid Closet was directed by Rob Epstein and Jeffrey Friedman.
  Fact 0.2: The Celluloid Closet was written by Rob Epstein and Jeffrey Friedman.
Sentence 1: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.
  Fact 1.0: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening".
  Fact 1.1: The film Station to Station is made up of 62 one-minute films.

Retrieved items:
<item 1>
Metro Station (band)
Metro Station is an American pop rock band that was formed in Los Angeles by Trace Cyrus and Mason Musso.

 In late 2006, the band signed a recording contract with Columbia Records and RED Ink Records.

 The band is best known for the Top 10 "Billboard" hit single "Shake It" from the group's self-titled debut album.

 In 2010, tension between Cyrus and Musso caused the band to go on hiatus.

 In 2011 the band returned, however it was announced that Cyrus was no longer a part of the group and Musso had purchased the rights to the name.

 An EP entitled "Middle of the Night" was released in 2013, which was led by the single "Every Time I Touch You".

 In 2014 Cyrus and a new drummer, Spencer Steffan came to the band and a new single entitled "Love & War" was released.

 In 2015, the band released a second full-length album titled "Savior".
</item 1>

<item 2>
Commit This to Memory
Commit This to Memory is the second studio album by American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on June 7, 2005, in the United States by Epitaph Records.

 Motion City Soundtrack, formed in 1997, had first found success with their debut album, "I Am the Movie" (2002).

 The band toured in the interim years, creating positive word-of-mouth.

 In 2004, the band joined Blink-182 on the road for a string of shows, which led to their bassist, Mark Hoppus, joining the band in the studio for his first
producing effort.
</item 2>

<item 3>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 3>

<item 4>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 4>

<item 5>
Graph query row:
film: The Celluloid Closet
release_date_or_year: 1995
relationship: None
connected_film: Station to Station
connected_release_date_or_year: 2014
</item 5>

<item 6>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 6>

<item 7>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 7>

<item 8>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 8>

<item 9>
The Boxer (Kele Okereke album)
The Boxer is the debut solo album by Kele Okereke, the lead singer of British indie rock band Bloc Party.

 Okereke released the album under the professional name of Kele on 21 June 2010.

 As promotion, he uploaded the songs "Rise" and "Walk Tall" to his personal website on 13 May.

 The first single from "The Boxer" was "Tenderoni", released on 14 June, and the second, "Everything You Wanted", on 16 August.

 The album's third single "On The Lam" was released on 25 October.
</item 9>

<item 10>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 10>
=== END ITEM id=1d1e5030aec462db ===

=== ITEM id=8587726f75a098b9 ===
Gold sentences and facts:
Sentence 0:  Her only novel, "The Bone People", won the Booker Prize in 1985.
  Fact 0.0: Her only novel is 'The Bone People'.
  Fact 0.1: 'The Bone People' won the Booker Prize.
  Fact 0.2: 'The Bone People' won the Booker Prize in 1985.
Sentence 1:  His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science fiction's top U.S. prizes in consecutive years.
  Fact 1.0: Card's novel 'Ender's Game' was published in 1985.
  Fact 1.1: 'Speaker for the Dead' is the sequel to 'Ender's Game'.
  Fact 1.2: 'Speaker for the Dead' was published in 1986.
  Fact 1.3: 'Ender's Game' won the Hugo Award.
  Fact 1.4: 'Ender's Game' won the Nebula Award.
  Fact 1.5: 'Speaker for the Dead' won the Hugo Award.
  Fact 1.6: 'Speaker for the Dead' won the Nebula Award.
  Fact 1.7: Card is the only author to win both science fiction's top U.S. prizes in consecutive years.

Retrieved items:
<item 1>
Ender in Exile
"Ender in Exile" begins one year after Ender has won the bugger war, and begins with the short story "Ender's Homecoming" from Card's webzine Intergalactic
Medicine Show.

 Other short stories that were published elsewhere are included as chapters of the novel.
</item 1>

<item 2>
Orson Scott Card (b. 1951), an American novelist and commentator, is best known for the Ender series. *Ender’s Game* (1985) and *Speaker for the Dead* (1986)
won Hugo and Nebula Awards in consecutive years. Related works include *Ender in Exile* (2008), *A War of Gifts* (2007), and *Shadow of the Hegemon* (2001); the
1977 “Ender
</item 2>

<item 3>
A War of Gifts: An Ender Story
A War of Gifts: An Ender Story (2007) is a science fiction novel by Orson Scott Card.

 This book is set in Card's Ender's Game series and takes place during Ender Wiggin's time at Battle School as described in Card's novels "Ender's Game" and
"Ender's Shadow".
</item 3>

<item 4>
Ender Wiggin
Andrew "Ender" Wiggin is a fictional character from Orson Scott Card's 1985 science fiction novel "Ender's Game" and its sequels ("Speaker for the Dead",
"Xenocide", "Children of the Mind", "Ender in Exile"), as well as in the first part of the spin-off series, "Ender's Shadow".

 The book series itself is an expansion, with some changes to detail, of Card's 1977 short story "Ender's Game."
</item 4>

<item 5>
Shadow of the Hegemon
Shadow of the Hegemon (2001) is the second novel in the "Ender's Shadow" series (often called the Bean Quartet) by Orson Scott Card.

 It is also the sixth novel in the "Ender's Game" series.

 It is told mostly from the point of view of Bean, a largely peripheral character in the original novel "Ender's Game" but the central protagonist of the
parallel narrative "Ender's Shadow."

 "Shadow of the Hegemon" was nominated for a Locus Award in 2002.
</item 5>

<item 6>
Ender in Exile
Ender in Exile is a science fiction novel by Orson Scott Card, part of the "Ender's Game" series, published on November 11, 2008.

 It takes place between the two award-winning novels: "Ender's Game" and "Speaker for the Dead".

 It could also be considered a parallel novel to the first three sequels in the Shadow Saga, since the entirety of this "trilogy" takes place in the span of
"Ender in Exile".

 The novel concludes a dangling story line of the Shadow Saga, while it makes several references to events that take place during the Shadow Saga.

 From yet another perspective, the novel expands (or "replaces") the last chapter of the original novel "Ender's Game".

 On the one hand, it fills the gap right before the last chapter, and on the other hand, it fills the gap between the last chapter and the original (first)
sequel (both named "Speaker for the Dead").
</item 6>

<item 7>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 7>

<item 8>
Ender's Game (short story)
"Ender's Game" is a story by Orson Scott Card.

 It first appeared in the August 1977 issue of "Analog" magazine and was later expanded into the novel "Ender's Game".

 Although the foundation of the Ender's Game series, the short story is not properly part of the "Ender's Game" universe, as there are many discrepancies in
continuity.
</item 8>

<item 9>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 9>

<item 10>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 10>
=== END ITEM id=8587726f75a098b9 ===

=== ITEM id=7f084802a98a5a0d ===
Gold sentences and facts:
Sentence 0: Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
  Fact 0.0: Giuseppe Fortunino Francesco Verdi was born on 9 or 10 October 1813.
  Fact 0.1: Giuseppe Fortunino Francesco Verdi died on 27 January 1901.
  Fact 0.2: Giuseppe Fortunino Francesco Verdi was an Italian opera composer.
Sentence 1: Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
  Fact 1.0: Charles Louis Ambroise Thomas was born on 5 August 1811.
  Fact 1.1: Charles Louis Ambroise Thomas died on 12 February 1896.
  Fact 1.2: Charles Louis Ambroise Thomas was a French composer.
  Fact 1.3: Charles Louis Ambroise Thomas is best known for his opera 'Mignon' (1866).
  Fact 1.4: Charles Louis Ambroise Thomas is best known for his opera 'Hamlet' (1868).
  Fact 1.5: The opera 'Hamlet' (1868) is after Shakespeare.
  Fact 1.6: Charles Louis Ambroise Thomas is best known as Director of the Conservatoire de Paris.
  Fact 1.7: Charles Louis Ambroise Thomas was Director of the Conservatoire de Paris from 1871 till his death.

Retrieved items:
<item 1>
Ferdinando Provesi
Ferdinando Angelo Maria Provesi (1770 – 1833) was a native of Parma, Italy.

 He was regarded as one of the greatest Italian opera composers of the era .

 Provesi is best known as being an early tutor of Giuseppe Verdi when he was the "Maestro di cappella" (master of music) at the St. Bartolomeo cathedral in
Busseto (the town very close to Le Roncole, the village where Verdi was born.)

 Provesi was also director of the municipal music school and local Philharmonic Society.

 He began teaching Verdi in 1824 when the future composer was 11 years old.
</item 1>

<item 2>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 2>

<item 3>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 3>

<item 4>
Ambroise Thomas
Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after
Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
</item 4>

<item 5>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 5>

<item 6>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 6>

<item 7>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 7>

<item 8>
List of major opera composers
This list provides a guide to opera composers, as determined by their presence on a majority of compiled lists of significant opera composers.

 (See the "Lists Consulted" section for full details.)

 The composers run from Jacopo Peri, who wrote the first ever opera in late 16th century Italy, to John Adams, one of the leading figures in the contemporary
operatic world.

 The brief accompanying notes offer an explanation as to why each composer has been considered major.

 Also included is a section about major women opera composers, compiled from the same lists.

 For an introduction to operatic history, see opera.

 The organisation of the list is by birthdate.
</item 8>

<item 9>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 9>

<item 10>
Teatro Giuseppe Verdi
Teatro Giuseppe Verdi (the Giuseppe Verdi Theatre) is a small opera house located in a wing of the Rocca dei Marchesi Pallavicino on the Piazza Giuseppe Verdi
in Busseto, Italy, a town closely associated with the life of the opera composer, Giuseppe Verdi.

 From the 13th century, the “rocca” or “fortress” was the family’s palace; it is now the city hall after being acquired by the municipality in 1856.

 The theatre opened on 15 August 1868 and seats 300.
</item 10>
=== END ITEM id=7f084802a98a5a0d ===

=== ITEM id=7489b8b84845680a ===
Gold sentences and facts:
Sentence 0:  The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.
  Fact 0.0: The first edition was written by J. Eric Holmes.
  Fact 0.1: The first edition was based on the original work of Gary Gygax and Dave Arneson.
Sentence 1: John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine, an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and science fiction.
  Fact 1.0: John Eric Holmes M.D. was born on February 16, 1930.
  Fact 1.1: John Eric Holmes M.D. died on March 20, 2010.
  Fact 1.2: John Eric Holmes M.D. was an associate professor of neurology at the University of Southern California School of Medicine.
  Fact 1.3: John Eric Holmes M.D. was an author and promoter of fantasy role-playing games.
  Fact 1.4: John Eric Holmes M.D. was a noted fan and enthusiast of Edgar Rice Burroughs.
  Fact 1.5: John Eric Holmes M.D. was an American writer of non-fiction, fantasy and science fiction.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Basic Set
The Dungeons & Dragons Basic Set is a set of rulebooks for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 First published in 1977, it saw a handful of revisions and reprintings.

 The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.

 Later editions were edited by Tom Moldvay, Dave Cook, and Frank Mentzer.
</item 1>

<item 2>
Dungeons &amp; Dragons Master Rules
Dungeons & Dragons Master Rules is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1985 as an expansion to the "Basic Set".
</item 2>

<item 3>
Dungeons &amp; Dragons Companion Set
The Dungeons & Dragons Companion Set is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1984 as an expansion to the "Dungeons & Dragons Basic Set".
</item 3>

<item 4>
John Eric Holmes
John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine,
an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and
science fiction.

 His father, Wilfred "Jasper" Holmes was also a writer of adventure stories under the pen name Alec Hudson.

 His writings appeared under his full name and under variants such as Eric Holmes and J. Eric Holmes and the pen name Sidney Leland.
</item 4>

<item 5>
The Keep on the Borderlands
The Keep on the Borderlands is a "Dungeons & Dragons" module by Gary Gygax, first printed in December 1979.

 In it, player characters are based at a keep and investigate a nearby series of caves that are filled with a variety of monsters.

 It was designed to be used with the "Dungeons & Dragons Basic Set", and was included in the 1979–1982 editions of the "Basic Set".

 It was designed for people new to "Dungeons & Dragons".
</item 5>

<item 6>
GURPS Basic Set
GURPS Basic Set is a hard-bound two volume set written by Steve Jackson, Sean M. Punch, and David L. Pulver.

 The first edition "GURPS Basic Set" was published in 1986.
</item 6>

<item 7>
Dungeons &amp; Dragons Immortals Rules
Dungeons & Dragons Immortals Rules, written by Frank Mentzer, is a boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game first published by
TSR in 1986 as an expansion to the "Basic Set".
</item 7>

<item 8>
Dungeons &amp; Dragons Game (1991 boxed set)
The New Easy-to-Master Dungeons & Dragons Game is an introductory set for the "Dungeons & Dragons" fantasy role-playing game, published by TSR, Inc. in 1991.

 It was a replacement for the previous "Dungeons & Dragons Basic Set", serving to introduce new players to the game, using the rule set previously established.
</item 8>

<item 9>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 9>

<item 10>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 10>
=== END ITEM id=7489b8b84845680a ===

=== ITEM id=6f48a25d2443f23b ===
Gold sentences and facts:
Sentence 0: Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.
  Fact 0.0: Haringey Heartlands is an urban regeneration project.
  Fact 0.1: Haringey Heartlands is currently ongoing.
  Fact 0.2: Haringey Heartlands is in the London Borough of Haringey.
  Fact 0.3: The London Borough of Haringey is in north London, England.
Sentence 1: The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.
  Fact 1.0: The Mall Wood Green is a large shopping centre.
  Fact 1.1: The Mall Wood Green is a residential complex.
  Fact 1.2: The Mall Wood Green is in Wood Green, north London.

Retrieved items:
<item 1>
The Mall Wood Green
The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.

 It is generally still referred to by its former name of Wood Green Shopping City; the signage on the building still uses that name.
</item 1>

<item 2>
White Hart Lane Community Sports Centre
White Hart Lane Community Sports Centre, also known as the New River Stadium is a rugby league and athletics stadium in Wood Green, Haringey, north London,
England that is home to London Skolars rugby league club, Enfield and Haringey Athletic Club, Wood Green Weightlifting Club, Hashtag United F.C. (YouTube based
football club), Haringey Rhinos rugby union club, Haringey Cycling Club, and the Next Level Football League.
</item 2>

<item 3>
London Skolars
London Skolars R.L.F.C. are a professional rugby league club based at the New River Stadium, Wood Green, Haringey in north London.

 They were founded in 1995 and have been professional since 2003, operating in the Kingstone Press League 1 in the 2015 season.

 They also run an A-team (formerly known as "Haringey Hornets") that play in the South Premier Division of the Rugby League Conference.
</item 3>

<item 4>
Haringey London Borough Council
Haringey London Borough Council is the local authority for the London Borough of Haringey in Greater London, England.

 It is a London borough council, one of 32 in the United Kingdom capital of London.

 Haringey is divided into 19 wards, each electing three councillors.

 Haringey London Borough Council comprises 48 Labour Party councillors, and 9 Liberal Democrats.

 The council was created by the London Government Act 1963 and replaced three local authorities: Hornsey Borough Council, Tottenham Borough Council and Wood
Green Borough Council.

 The next election to the authority will be in 2018.
</item 4>

<item 5>
Wood Green Town F.C.
Wood Green Town Football Club was a football club based in Tottenham in the London Borough of Haringey, England.

 Formed in 1911 as a breakaway from Tufnell Park, they merged with Edmonton (a descendent of Tufnell Park) in 1973 to form Edmonton & Haringey.

 During their history the club were known as Tufnell Spartans, Wood Green, Wood Green Town and Haringey Borough.
</item 5>

<item 6>
Haringey Heartlands
Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.

 The site is located west of Shopping City in Wood Green.

 The aim of the development is "to create a vibrant and attractive new urban quarter which acts as the civic and cultural Heart of Haringey, integrating with
and benefiting wider communities."
</item 6>

<item 7>
Alexandra Palace railway station
Alexandra Palace railway station (originally named Wood Green and later Wood Green (Alexandra Park)) is in the London Borough of Haringey in north London, and
is in Zone 3.

 The station, and all trains serving it, are operated by Great Northern.

 Trains mostly go to and from Moorgate on weekdays and weekends.
</item 7>

<item 8>
Heartlands High School
Heartlands High School is a state secondary school located in Wood Green in the London Borough of Haringey.

 It welcomed its first students in September 2010.

 The headteacher is Mr. Simon Garrill.
</item 8>

<item 9>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 9>

<item 10>
Wood Green tube station
Wood Green is a London Underground station on the Piccadilly line.

 The station is between Turnpike Lane and Bounds Green stations and is in Travelcard Zone 3.

 It is located at junction of High Road, Wood Green and Lordship Lane.

 It serves Wood Green Shopping City and the nearby Haringey Council administrative complex as well as a densely populated residential area.

 It is also the closest tube station to the Alexandra Palace Venue.
</item 10>
=== END ITEM id=6f48a25d2443f23b ===

=== ITEM id=fb4e021fedd7dc97 ===
Gold sentences and facts:
Sentence 0: Eryngium is a genus of flowering plants in the family Apiaceae.
  Fact 0.0: Eryngium is a genus of flowering plants.
  Fact 0.1: Eryngium is in the family Apiaceae.
Sentence 1: Arnebia is a genus of flowering plants in the family Boraginaceae.
  Fact 1.0: Arnebia is a genus of flowering plants.
  Fact 1.1: Arnebia is in the family Boraginaceae.

Retrieved items:
<item 1>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 1>

<item 2>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 2>

<item 3>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 3>

<item 4>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 4>

<item 5>
Cyanobiont
Several cyanobionts involved with fungi and marine organisms also belong to the genera "Richelia, Calothrix, Synechocystis, Aphanocapsa" and "Anabaena", as well
as the species "Oscillatoria spongeliae".

 Although there are many documented symbioses between cyanobacteria and marine organisms, little is known about the nature of many of these symbioses.

 The possibility of discovering more novel symbiotic relationships is apparent from preliminary microscopic observations.
</item 5>

<item 6>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 6>

<item 7>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 7>

<item 8>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 8>

<item 9>
Extraterrestrial life
Extraterrestrial life, also called alien life (or, if it is a sentient or relatively complex individual, an "extraterrestrial" or "alien"), is life that does
not originate from Earth.

 These hypothetical life forms may range from simple prokaryotes to beings with civilizations far more advanced than humanity.

 The Drake equation speculates about the existence of intelligent life elsewhere in the universe.

 The science of extraterrestrial life in all its forms is known as exobiology.
</item 9>

<item 10>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 10>
=== END ITEM id=fb4e021fedd7dc97 ===

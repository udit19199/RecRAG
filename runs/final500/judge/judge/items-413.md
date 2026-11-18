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

=== ITEM id=9b49a3e5658930ee ===
Gold sentences and facts:
Sentence 0:  He has been involved in numerous television projects as director, writer, producer and animator.
  Fact 0.0: The person described has been involved in numerous television projects.
  Fact 0.1: The person described has worked on television projects as a director.
  Fact 0.2: The person described has worked on television projects as a writer.
  Fact 0.3: The person described has worked on television projects as a producer.
  Fact 0.4: The person described has worked on television projects as an animator.
Sentence 1: Béla Gaál (2 January 1893 – 18 February 1945) was a Hungarian film director.
  Fact 1.0: Bela Gaal was born on 2 January 1893.
  Fact 1.1: Bela Gaal died on 18 February 1945.
  Fact 1.2: Bela Gaal was a Hungarian film director.

Retrieved items:
<item 1>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 1>

<item 2>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 2>

<item 3>
Béla Gaál
Béla Gaál (2 January 1893 – 18 February 1945) was a Hungarian film director.

 His 1930 film "Csak egy kislány van a világon" was the first sound film to be made in Hungary.
</item 3>

<item 4>
Heavy Traffic
Heavy Traffic is a 1973 American adult animated comedy-drama film written and directed by Ralph Bakshi.

 The film, which begins, ends, and occasionally combines with live-action, explores the often surreal fantasies of a young New York cartoonist named Michael
Corleone, using pinball imagery as a metaphor for inner-city life.

 "Heavy Traffic" was Bakshi and producer Steve Krantz's follow-up to the film "Fritz the Cat".

 Though producer Krantz made varied attempts to produce an R-rated film, "Heavy Traffic" was given an X rating by the MPAA.

 The film received positive reviews and is widely considered to be Bakshi's biggest critical success.
</item 4>

<item 5>
Radha Bharadwaj
Radha Bharadwaj is an Indian filmmaker, film producer and screenwriter.

 She moved to the United States in her late teens to study film.

 Bharadwaj's screenwriting and directing feature debut is "Closet Land".

 The surreal psychological drama was released by Universal Pictures in 1991, making Radha Bharadwaj the first director of Indian descent to have a film released
by a major Hollywood studio.

 "Closet Land" stars Alan Rickman and Madeleine Stowe.

 Ron Howard and Brian Grazer produced the feature.

 The screenplay for "Closet Land" won the Nicholl Screenwriting Fellowship sponsored by the Academy of Motion Picture Arts and Sciences, among others.
</item 5>

<item 6>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 6>

<item 7>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
</item 7>

<item 8>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 8>

<item 9>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 9>

<item 10>
Tales of Budapest
Tales Of Budapest (Hungarian: "Pesti mese" ) is a 1937 Hungarian comedy film directed by Béla Gaál.
</item 10>
=== END ITEM id=9b49a3e5658930ee ===

=== ITEM id=c20c9bc633676001 ===
Gold sentences and facts:
Sentence 0: Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11, 1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical physics.
  Fact 0.0: Nikolay Mitrofanovich Krylov was born on 29 November [O.S. 17 November] 1879.
  Fact 0.1: Nikolay Mitrofanovich Krylov died on May 11, 1955.
  Fact 0.2: Nikolay Mitrofanovich Krylov's Russian name is Николай Митрофанович Крылов.
  Fact 0.3: Nikolay Mitrofanovich Krylov's Ukrainian name is Микола Митрофанович Крилов.
  Fact 0.4: Nikolay Mitrofanovich Krylov was a Russian mathematician.
  Fact 0.5: Nikolay Mitrofanovich Krylov was a Soviet mathematician.
  Fact 0.6: Nikolay Mitrofanovich Krylov is known for works on interpolation.
  Fact 0.7: Nikolay Mitrofanovich Krylov is known for works on non-linear mechanics.
  Fact 0.8: Nikolay Mitrofanovich Krylov is known for works on numerical methods for solving equations of mathematical physics.
Sentence 1: Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.
  Fact 1.0: Anatoly Timofeevich Fomenko's Russian name is Анатолий Тимофеевич Фоменко.
  Fact 1.1: Anatoly Timofeevich Fomenko was born on 13 March 1945 in Stalino, USSR.
  Fact 1.2: Anatoly Timofeevich Fomenko is a Soviet mathematician.
  Fact 1.3: Anatoly Timofeevich Fomenko is a Russian mathematician.
  Fact 1.4: Anatoly Timofeevich Fomenko is a professor at Moscow State University.
  Fact 1.5: Anatoly Timofeevich Fomenko is well known as a topologist.
  Fact 1.6: Anatoly Timofeevich Fomenko is a member of the Russian Academy of Sciences.

Retrieved items:
<item 1>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 1>

<item 2>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 2>

<item 3>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 3>

<item 4>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 4>

<item 5>
Vladimir Semenikhin
Vladimir Anatolievich Semenikhin (born August 31, 1967) is a Russian businessman, philanthropist, art collector and honorary consul.

 Semenikhin is the president of the Ekaterina Cultural Foundation, Chairman of the board of directors at Stroyteks Group of companies and the Honorary Consul of
Kazakhstan in the Principality of Monaco.
</item 5>

<item 6>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 6>

<item 7>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 7>

<item 8>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 8>

<item 9>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 9>

<item 10>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 10>
=== END ITEM id=c20c9bc633676001 ===

=== ITEM id=37639a331f511347 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province, People's Republic of China.
Sentence 1: Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.
  Fact 1.0: Hailar District's Mongolian name is ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ, romanized as Qayilar toɣoriɣ.
  Fact 1.1: Hailar District's Mongolian name in Cyrillic script is Хайлаар тойрог.
  Fact 1.2: Hailar District's Cyrillic name is Хайлаар дүүрэг, romanized as Khailaar düüreg.
  Fact 1.3: Hailar District is an urban district.
  Fact 1.4: Hailar District serves as the seat of the prefecture-level city Hulunbuir.
  Fact 1.5: Hailar District is located in northeastern Inner Mongolia, China.

Retrieved items:
<item 1>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 1>

<item 2>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 2>

<item 3>
Old Barag Banner
Old Barag Banner (Mongolian: ᠬᠠᠭᠤᠴᠢᠨ ᠪᠠᠷᠭᠤ ᠬᠣᠰᠢᠭᠤ Қаучин Бару қосиу "Qaɣučin Barɣu qosiɣu"; ) is a banner of northeastern Inner Mongolia, People's Republic of
China.

 It is located 28 km from Hailar District, the administrative centre of Hulunbuir City, which administers this banner.
</item 3>

<item 4>
Hailar District
Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as
the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.

 Hulunbuir, due to its massive size, is a city in administrative terms only, being mainly grassland and rural.

 Hailar, therefore, is a "de facto" city.

 Hailar can also refer to the urban area around the Hailar district, with Hulunbuir being the wider geographical region that contains the urban area.
</item 4>

<item 5>
Hulunbuir
Hulunbuir or Hulun Buir (Mongolian: ; , "Hūlúnbèi'ěr") is a region that is governed as a prefecture-level city in northeastern Inner Mongolia, in China.

 Its administrative center is located at Hailar District, its largest urban area.

 Major scenic features are the high steppes of the Hulun Buir grasslands, the Hulun and Buir lakes (the latter partially in Mongolia), and the Khingan range.

 Hulun Buir borders Russia to the north and west, Mongolia to the south and west, Heilongjiang province to the east and Hinggan League to the direct south.

 Hulunbuir is a linguistically diverse area: next to Mandarin Chinese, Mongolian dialects such as Khorchin and Buryat, the Mongolic language Dagur and some
Tungusic languages are spoken there.
</item 5>

<item 6>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 6>

<item 7>
Entries describe Chinese and Korean places, people, and organizations: Pungdo is an inhabited Yellow Sea island in Ansan, South Korea. Liaoning locations
include Pulandian (2,770 km²; 741,230 people), Longkou (620,000), Kaiyuan, Qinghe District, Shuangta, Taiyang, Xingcheng, and Kaiyuan West station. Politicians
Du Zheheng (1910–75), Liaoning’s first governor
</item 7>

<item 8>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 8>

<item 9>
The entries cover Karakoram peaks and subranges, mostly in Gilgit-Baltistan, Pakistan, and Xinjiang, China. Hispar Muztagh’s highest peak is Distaghil Sar
(7,885 m), among the world’s highest; it has three summits. Other notable peaks include Kanjut Sar, Momhil Sar, Yutmaru Sar, Passu Sar, Mingli Sar (first
climbed by Nazir Sabir in 1988), Kez
</item 9>

<item 10>
Old Barag Banner lies 28 km from Hailar, Hulunbuir’s administrative center in northeastern Inner Mongolia. Hulunbuir is a vast, linguistically diverse region
bordering Russia, Mongolia, Heilongjiang and Hinggan League; Hailar is its practical urban center. Fuzhou Metro Line 6 is a planned 40.9-km airport route. Other
entries describe a Ningbo metro station; Australian native-vegetation clearing; the
</item 10>
=== END ITEM id=37639a331f511347 ===

=== ITEM id=f3fe3d0ebb88aaa5 ===
Gold sentences and facts:
Sentence 0: Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.
  Fact 0.0: Ghost Rider is a song.
  Fact 0.1: Ghost Rider is a song by the protopunk band Suicide.
  Fact 0.2: Ghost Rider appears on the debut album of the protopunk band Suicide.
Sentence 1:  The song is based on the Marvel Comics character.
  Fact 1.0: The song Ghost Rider is based on the Marvel Comics character.
Sentence 2: Suicide is the debut album from the American rock band Suicide.
  Fact 2.0: The album Suicide is the debut album of the American rock band Suicide.
  Fact 2.1: The band Suicide is an American rock band.
Sentence 3:  It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.
  Fact 3.0: The album Suicide was released in 1977.
  Fact 3.1: The album Suicide was released on Red Star Records.
  Fact 3.2: The album Suicide was produced by Craig Leon.
  Fact 3.3: The album Suicide was produced by Marty Thau.

Retrieved items:
<item 1>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 1>

<item 2>
Ghost Rider (Suicide song)
Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.

 The song is based on the Marvel Comics character.
</item 2>

<item 3>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 3>

<item 4>
The Aeroplane Flies High
The Aeroplane Flies High is a five-disc box set released by American alternative rock band The Smashing Pumpkins in 1996.

 It contains expanded versions of the five singles from their album "Mellon Collie and the Infinite Sadness" and also included a 44-page booklet with pictures
and writings by the band's lead singer Billy Corgan, as well as lyrics.

 A limited edition release, the box reached number 42 on the "Billboard" charts, and sold 300,000 units (1.5 million discs in all), generating a platinum disc
for the band.

 Originally intended to be limited to 200,000 copies, Virgin Records produced more after the original run sold out due to overwhelming and unexpected demand.

 The album was remastered in 2013 under the supervision of frontman Billy Corgan and reissued on vinyl and as a CD/DVD box set.
</item 4>

<item 5>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 5>

<item 6>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 6>

<item 7>
It's Superman!
It's Superman!

 is a novel by Tom De Haven based on the comic book superhero Superman.

 It was released on September 15, 2005 in hardcover and August 29, 2006 in paperback.

 The premise tells the tale of Clark Kent's beginnings into becoming Superman, set in the 1930s, where Clark befriends a wrongly convicted photographer named
Willi Berg, and is then taken from Kansas to Hollywood and finally in New York where he meets Lois Lane, fights Lex Luthor, as he debuts in his superhero
persona.

 Despite the setting, this is not about the Golden Age Superman also known as the Superman of Earth-2; as Perry White, the Daily Planet, Lex Luthor's position
and his trademark powers (including flight) are not part of that alternate Earth.

 Rather, it's a Superman period piece set in the 1930s.
</item 7>

<item 8>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 8>

<item 9>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 9>

<item 10>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 10>
=== END ITEM id=f3fe3d0ebb88aaa5 ===

=== ITEM id=7540fb3e377e32de ===
Gold sentences and facts:
Sentence 0: Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.
  Fact 0.0: Brett Dallas McClure was born on February 19, 1981.
  Fact 0.1: Brett Dallas McClure was born in Yakima, Washington.
  Fact 0.2: Brett Dallas McClure is a retired American gymnast.
Sentence 1:  He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the team competition at the 2004 Summer Olympics in Athens.
  Fact 1.0: Brett Dallas McClure won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia.
  Fact 1.1: Brett Dallas McClure later helped his U.S. gymnastics team earn a silver medal in the team competition at the 2004 Summer Olympics in Athens.
Sentence 2:  The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.
  Fact 2.0: The competition was held in Brisbane, Queensland, Australia.
  Fact 2.1: The competition was held from 29 August to 9 September 2001.

Retrieved items:
<item 1>
Anthony Biddle (Paralympian)
Anthony John Biddle, OAM (born 18 June 1975) is an Australian Paralympic tandem cyclist and athlete.

 He was born in the New South Wales city of Gosford.

 He competed in athletics without winning any medals at the 1996 Atlanta Games and the 2000 Sydney Games.

 At the 2004 Athens Games, he switched to cycling and won a gold medal in the Men's 1 km Time Trial Tandem B1–3 event, for which he received a Medal of the
Order of Australia, and a bronze medal in the Men's Sprint Tandem B1–3 event.

 Kial Stewart was his pilot for both events.
</item 1>

<item 2>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 2>

<item 3>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 3>

<item 4>
Brett McClure
Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.

 He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the
team competition at the 2004 Summer Olympics in Athens.

 During his sporting career, McClure has collected two more silver medals in the same program at the World Artistic Gymnastics Championships (2001 and 2003).

 McClure is currently the High Performance Director with USA men's gymnastics.
</item 4>

<item 5>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 5>

<item 6>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 6>

<item 7>
Nicolas N’Godrela won three tennis golds at the 2015 Pacific Games. Aerobic gymnast Remus Nicolai earned 11 world and European medals, then coached in Romania.
Brett McClure won Olympic team silver and is USA Gymnastics’ men’s High Performance Director. Masayoshi Yamamoto won 2014 Asian Games pommel horse gold. Louis
Smith won five Olympic medals across three Games and a 2015 European pommel horse title. Luke
</item 7>

<item 8>
Guadalajara’s 2011 Pan American Games hosted 6,000 athletes from 42 nations; Toronto’s 2015 Games spanned 18 communities. Port Moresby’s 2015 Pacific Games
included Australia’s debut: Papua New Guinea led women’s powerlifting, Nauru men’s, and Tuvalu won its first gold; Nicolas N’Godrela earned three tennis golds.
Profiles span footballers, champion jockeys (Tony McCoy: 4
</item 8>

<item 9>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 9>

<item 10>
The 2011 Pan American Games in Guadalajara involved about 6,000 athletes from 42 nations; Toronto hosted the 2015 edition across 18 Golden Horseshoe
communities. Port Moresby’s 2015 Pacific Games featured Kiribati, Australia’s debut, and events including basketball, football and powerlifting. Papua New
Guinea dominated women’s powerlifting; Nauru led the men’s, while Tuvalu earned its first Pacific Games gold. Vanuatu
</item 10>
=== END ITEM id=7540fb3e377e32de ===

=== ITEM id=957242d73333063d ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province, People's Republic of China.
Sentence 1: Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.
  Fact 1.0: Hailar District's Mongolian name is ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ, romanized as Qayilar toɣoriɣ.
  Fact 1.1: Hailar District's Mongolian name in Cyrillic script is Хайлаар тойрог.
  Fact 1.2: Hailar District's Cyrillic name is Хайлаар дүүрэг, romanized as Khailaar düüreg.
  Fact 1.3: Hailar District is an urban district.
  Fact 1.4: Hailar District serves as the seat of the prefecture-level city Hulunbuir.
  Fact 1.5: Hailar District is located in northeastern Inner Mongolia, China.

Retrieved items:
<item 1>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 1>

<item 2>
Hailar District
Hailar District (Mongolian: ᠬᠠᠶᠢᠯᠠᠷ ᠲᠣᠭᠣᠷᠢᠭ "Qayilar toɣoriɣ" Хайлаар тойрог, Cyrillic: Хайлаар дүүрэг, "Khailaar düüreg"; ) is an urban district that serves as
the seat of the prefecture-level city Hulunbuir in northeastern Inner Mongolia, China.

 Hulunbuir, due to its massive size, is a city in administrative terms only, being mainly grassland and rural.

 Hailar, therefore, is a "de facto" city.

 Hailar can also refer to the urban area around the Hailar district, with Hulunbuir being the wider geographical region that contains the urban area.
</item 2>

<item 3>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 3>

<item 4>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 4>

<item 5>
Old Barag Banner
Old Barag Banner (Mongolian: ᠬᠠᠭᠤᠴᠢᠨ ᠪᠠᠷᠭᠤ ᠬᠣᠰᠢᠭᠤ Қаучин Бару қосиу "Qaɣučin Barɣu qosiɣu"; ) is a banner of northeastern Inner Mongolia, People's Republic of
China.

 It is located 28 km from Hailar District, the administrative centre of Hulunbuir City, which administers this banner.
</item 5>

<item 6>
Ludhiana
Ludhiana is a city and a municipal corporation in Ludhiana district in the Indian state of Punjab, and is the largest city north of Delhi.

 It is the largest city in the state, with an area of 310sq.

 km and an estimated population of 1,789,650 as of the 2015.

 The population increases substantially during the harvesting season due to the migration of labourers from the eastern states of Uttar Pradesh, Bihar, Odisha.

 The city stands on the Sutlej River's old bank, 13 km south of its present course.

 It is an industrial centre of northern India, and was referred to as India's Manchester by the BBC.
</item 6>

<item 7>
Hulunbuir
Hulunbuir or Hulun Buir (Mongolian: ; , "Hūlúnbèi'ěr") is a region that is governed as a prefecture-level city in northeastern Inner Mongolia, in China.

 Its administrative center is located at Hailar District, its largest urban area.

 Major scenic features are the high steppes of the Hulun Buir grasslands, the Hulun and Buir lakes (the latter partially in Mongolia), and the Khingan range.

 Hulun Buir borders Russia to the north and west, Mongolia to the south and west, Heilongjiang province to the east and Hinggan League to the direct south.

 Hulunbuir is a linguistically diverse area: next to Mandarin Chinese, Mongolian dialects such as Khorchin and Buryat, the Mongolic language Dagur and some
Tungusic languages are spoken there.
</item 7>

<item 8>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 8>

<item 9>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 9>

<item 10>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 10>
=== END ITEM id=957242d73333063d ===

=== ITEM id=729499c14a3ef7ae ===
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
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 1>

<item 2>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 2>

<item 3>
Paul Scott (novelist)
Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."

 His novel "Staying On" won the Booker Prize for 1977.
</item 3>

<item 4>
The Killer (play)
The Killer (French: Tueur sans gages , sometimes translated "The Killer without Reason" or "The Killer without Cause") is a play written by Eugène Ionesco in
1958.

 It is the first of Ionesco's Berenger plays, the others being "Rhinocéros" (1959), "Exit the King" (1962), and "A Stroll in the Air" (1963).
</item 4>

<item 5>
Lonely Planet (play)
Lonely Planet is a two character play written by Steven Dietz.

 The play tells the story of Jody and Carl, two gay men who live in an unnamed American city.

 The play was written during the midst of the AIDS epidemic, which is the central focus of the story, though ultimately the play sends the message that one
should pay attention to the world around them and realize its problems rather than shun it.

 The play heavily references the Eugène Ionesco comedy "The Chairs".
</item 5>

<item 6>
Exit the King
Exit the King (French: Le Roi se meurt ) is an absurdist drama by Eugène Ionesco that premiered in 1962.

 It is the third in Ionesco's "Berenger Cycle", preceded by "The Killer" (1958) and "Rhinocéros" (1959), and followed by "A Stroll in the Air" (1963).
</item 6>

<item 7>
The New Tenant
The New Tenant (French: "Le Nouveau Locataire" ) is a play written by Eugène Ionesco in 1955.

 The central image is common to many Ionesco plays: something accumulates on stage and overwhelms the characters.

 In this case its furniture.

 The main characters are a gentleman, a caretaker, and two movers.

 The caretaker talks as the gentleman, the "new tenant" of the title, directs the two movers who continuously bring in furniture.
</item 7>

<item 8>
Eugène Ionesco
Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost
figures of the French Avant-garde theatre.

 Beyond ridiculing the most banal situations, Ionesco's plays depict the solitude and insignificance of human existence in a tangible way.
</item 8>

<item 9>
Victims of Duty
Victims of Duty (French: Victimes du Devoir ) is a one-act play written in 1953 by French-Romanian playwright Eugène Ionesco.

 An early work, it has not received the notoriety of his other works.

 The play is in the Theatre of the Absurd style, of which Ionesco was a pioneer.
</item 9>

<item 10>
The Lesson
The Lesson (French: "La Leçon" ) is a one-act play by French-Romanian playwright Eugène Ionesco.

 It was first performed in 1951 in a production directed by Marcel Cuvelier (who also played the Professor).

 Since 1957 it has been in permanent showing at Paris' Théâtre de la Huchette, on an Ionesco double-bill with The Bald Soprano.

 The play is regarded as an important work in the "Theatre of the Absurd".
</item 10>
=== END ITEM id=729499c14a3ef7ae ===

=== ITEM id=66208854cf5c7dc3 ===
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
Trace Cyrus
Trace Dempsey Cyrus (born Neil Timothy Helson; February 24, 1989) is an American musician.

 The son of country singer Billy Ray Cyrus and brother of recording artist Miley Cyrus, he is the backing vocalist and bassist of the band Metro Station.

 He left the band in 2010 but returned in 2014.

 In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.

 He also owns the clothing company From Backseats to Bedrooms.
</item 1>

<item 2>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 2>

<item 3>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 3>

<item 4>
Brother Clyde (album)
Brother Clyde is the debut studio album from alternative rock group Brother Clyde.

 It is the first side-project from country music singer Billy Ray Cyrus.

 The album features duets with Cyrus' son Trace Cyrus as well as Dolly Parton, who Cyrus said "wanted to rock."

 It was released on August 10, 2010.

 Its first single, "Lately", which features rapper King Phaze, was issued to digital retailers on June 29, 2010, and a music video, which was directed by Cyrus,
was released for the song in July 2010.
</item 4>

<item 5>
Shrinking Violet (album)
Shrinking Violet is the sixth album by glam metal band L.A. Guns, first released on June 1, 1999, through Perris Records, and is the only L.A. Guns album with
singer Jizzy Pearl.

 The album was reissued, with bonus tracks and new artwork, on May 24, 2010, through Favored Nations.

 The album was produced by former Guns N' Roses guitarist Gilby Clarke who also provided additional guitars on the track "Dreamtime".
</item 5>

<item 6>
- **Quiet Riot 10** (2014), the band’s twelfth studio album and first since reuniting in 2010, features Jizzy Pearl on six new studio tracks; its final four
tracks are live performances with Kevin DuBrow, who died in 2007.
- **Inner Secrets** (1978), Santana’s ninth album, shifted toward album-oriented rock. “Stormy” and “One Chain” were hits; “Well All Right” reached No.
</item 6>

<item 7>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 7>

<item 8>
The entries profile rock, punk, and metal albums, bands, and musicians. Fuel’s *Puppet Strings* (2014) reunited singer Brett Scallions but featured none of the
prior album’s recording personnel. Jizzy Pearl sang on Quiet Riot’s *10* and L.A. Guns’ *Shrinking Violet*. Other subjects include J. Geils Band, Green Day side
projects and collaborators, Glassjaw and Daryl Palumbo, Clearview 77, Sons of Abraham,
</item 8>

<item 9>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 9>

<item 10>
Jizzy Pearl
Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.

 He's had notable performances in several heavy metal and hard rock groups.

 He first fronted the band Data Clan, which eventually became Love/Hate.

 Pearl has also sung for L.A. Guns, Ratt, Adler's Appetite, and other, lesser known acts.
</item 10>
=== END ITEM id=66208854cf5c7dc3 ===

=== ITEM id=bf140e13652ed1bb ===
Gold sentences and facts:
Sentence 0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.
  Fact 0.0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen.
  Fact 0.1: Not Richard, But Dick was released in 1993.
  Fact 0.2: Not Richard, But Dick was released by Hollywood Records.
Sentence 1: The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.
  Fact 1.0: The Dead Milkmen are an American punk rock band.
  Fact 1.1: The Dead Milkmen were formed in 1983.
  Fact 1.2: The Dead Milkmen were formed in Philadelphia, Pennsylvania.

Retrieved items:
<item 1>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 1>

<item 2>
Richard Hawley
Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.

 After his first band Treebound Story (formed while he was still at school) broke up, Hawley found success as a member of Britpop band Longpigs in the 1990s.

 After that group broke up in 2000, he later joined the band Pulp, led by his friend Jarvis Cocker, for a short time.

 As a solo musician, Hawley has released seven studio albums.

 He has been nominated for a Mercury prize twice and once for a Brit Award.

 He has collaborated with Lisa Marie Presley, Arctic Monkeys, Manic Street Preachers and Paul Weller.
</item 2>

<item 3>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 3>

<item 4>
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
</item 4>

<item 5>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 5>

<item 6>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 6>

<item 7>
Dick Delicious and The Tasty Testicles
Dick Delicious and The Tasty Testicles are a comedy / metal band from Atlanta, Georgia.

 The band formed in 1992 and released 3 albums between 1995 and 2003.

 In 2003 the Dick Delicious received the "Howard Stern Award for Musical Excellence" and broke up shortly afterward.
</item 7>

<item 8>
Graph query row:
album: Not Richard, But Dick
band: The Dead Milkmen
city: Philadelphia
</item 8>

<item 9>
Not Richard, But Dick
Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.

 After being out of print for years, Hollywood Records decided to release the album for digital download as of April 2, 2013.
</item 9>

<item 10>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 10>
=== END ITEM id=bf140e13652ed1bb ===

=== ITEM id=5c3c1a6380d50288 ===
Gold sentences and facts:
Sentence 0: Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.
  Fact 0.0: Kim Dong-jun's Hangul name is 김동준.
  Fact 0.1: Kim Dong-jun's Hanja name is 金桐俊.
  Fact 0.2: Kim Dong-jun was born February 11, 1992.
  Fact 0.3: Kim Dong-jun is a South Korean singer.
  Fact 0.4: Kim Dong-jun is a South Korean actor.
Sentence 1: Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.
  Fact 1.0: Charlie Harper was born David Charles Perez.
  Fact 1.1: Charlie Harper was born on 25 May 1944.
  Fact 1.2: Charlie Harper was born in Hackney, London.
  Fact 1.3: Charlie Harper is a British singer.
  Fact 1.4: Charlie Harper is a British songwriter.

Retrieved items:
<item 1>
Charlie Harper (singer)
Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.

 He is best known as the lead singer from the punk band U.K. Subs.
</item 1>

<item 2>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 2>

<item 3>
Kim Dong-jun (footballer)
Kim Dong-jun (Korean: 김동준 , born 19 December 1994) is a South Korean footballer who plays as a goalkeeper for Yonsei University and South Korea national team.
</item 3>

<item 4>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 4>

<item 5>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 5>

<item 6>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 6>

<item 7>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 7>

<item 8>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 8>

<item 9>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 9>

<item 10>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 10>
=== END ITEM id=5c3c1a6380d50288 ===

=== ITEM id=b2f0e6e2a7698c58 ===
Gold sentences and facts:
Sentence 0:  The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI Studios in London, which was soon renamed Abbey Road Studios.
  Fact 0.0: The title and cover are an homage to the Beatles album Abbey Road.
  Fact 0.1: 926 East McLemore Avenue is the address of the Stax Studios in Memphis.
  Fact 0.2: Abbey Road was the address of EMI Studios in London.
  Fact 0.3: EMI Studios in London was soon renamed Abbey Road Studios.
Sentence 1:  This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
  Fact 1.0: This road is best known for the Abbey Road Studios.
  Fact 1.1: This road is best known for the 1969 album Abbey Road by The Beatles.

Retrieved items:
<item 1>
List of roads named after Mahatma Gandhi
Mahatma Gandhi Road which is also used in its abbreviated form as M. G. Road, named after Mahatma Gandhi.

 M. G. Road is one of the most frequently used road names in India, Agra having two of them.

 Several other Indian cities have a road by this name.

 In 2010, attempts were made to rename Hillcroft Avenue in Houston, Texas to Mahatma Gandhi Avenue but the proponents were unable to secure signatures from 75%
of the property owners.

 As a result, the group settled for renaming the areas around Hillcroft Avenue to Mahatma Gandhi district.

 In Kolkata, Central Road was renamed Harrison Road in 1892 after Sir Henry Leland Harrison, then the chairman of Calcutta Corporation.

 It has recently been renamed yet again to Mahatma Gandhi Road.

 In many cities in South India, Mahatma Gandhi Road is popularly known by other names.

 In Chennai, it is known as Nungambakkam High Road.

 In Coimbatore, Mahatma Gandhi Road is popularly known by Avarampalayam Road.
</item 1>

<item 2>
Westminster
Westminster ( ) is an area of central London within the City of Westminster, part of the West End, on the north bank of the River Thames.

 Westminster's concentration of visitor attractions and historic landmarks, one of the highest in London, includes the Palace of Westminster, Buckingham Palace,
Westminster Abbey and Westminster Cathedral.
</item 2>

<item 3>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 3>

<item 4>
Temple Bar, London
Temple Bar was the principal ceremonial entrance to the City of London on its western side from the City of Westminster.

 It is situated on the historic royal ceremonial route from the Tower of London to the Palace of Westminster, the two chief residences of the mediaeval English
monarchs, and from the Palace of Westminster to St Paul's Cathedral.

 The road east of Temple Bar and within the City is Fleet Street, the road to the west, in Westminster, is The Strand.

 At Temple Bar the Corporation of the City of London formerly erected a barrier to regulate trade into the City.

 The 19th century Royal Courts of Justice are located next to it on its north side, having been moved from Westminster Hall.

 To its south is the Temple Church and the Inner Temple and Middle Temple Inns of Court.

 As the most important entrance to the City of London from Westminster, it was formerly long the custom for the monarch to halt at Temple Bar before entering
the City of London, in order for the Lord Mayor to offer the Corporation's pearl-encrusted Sword of State as a token of loyalty.
</item 4>

<item 5>
Westminster tube station
Westminster is a London Underground station in the City of Westminster.

 It is served by the Circle, District and Jubilee lines.

 On the Circle and District lines, the station is between St. James's Park and Embankment, and on the Jubilee line it is between Green Park and Waterloo.

 It is in Travelcard Zone 1.

 The station is located at the corner of Bridge Street and Victoria Embankment and is close to the Houses of Parliament, Westminster Abbey, Parliament Square,
Whitehall, Westminster Bridge, and the London Eye.

 Also close by are Downing Street, the Cenotaph, Westminster Millennium Pier, the Treasury, the Foreign and Commonwealth Office, and the Supreme Court.
</item 5>

<item 6>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 6>

<item 7>
City Cruises runs year-round, wheelchair-accessible Thames sightseeing services between major London piers, plus private charters, the London Showboat dinner
cruise and permanently moored Hispaniola restaurant. The remaining entries describe London locations and institutions: Doughty Street Chambers’ legal work and
Doughty Street’s history; Westminster’s landmarks and former parishes; Abbey Road and its Beatles association; Westminster Underground station; Temple Bar’s
ceremonial role and Wren gateway; and the former Westminster Magistrates’
</item 7>

<item 8>
City of Westminster Magistrates' Court
The City of Westminster Magistrates' Court was a magistrates' court located at 70 Horseferry Road, in the City of Westminster, London.

 It was originally called Horseferry Road Magistrates' Court, after the road in which it was sited.

 However, it was renamed in July 2006 following the closure of Bow Street Magistrates' Court.

 It served as the court where the Chief Magistrate of England and Wales sat, and all extradition and terrorism-related cases passed through the court.

 The court closed permanently on 22 September 2011, and was replaced on 27 September 2011 with Westminster Magistrates' Court, built on the site of Marylebone
Magistrates' Court at 181 Marylebone Road.
</item 8>

<item 9>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 9>

<item 10>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 10>
=== END ITEM id=b2f0e6e2a7698c58 ===

=== ITEM id=0ddfefcd3fb33835 ===
Gold sentences and facts:
Sentence 0: "White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".
  Fact 0.0: 'White People Renovating Houses' is the first episode of the twenty-first season of the American animated television series South Park.
Sentence 1:  It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
  Fact 1.0: 'White People Renovating Houses' is the 278th episode of South Park overall.
  Fact 1.1: 'White People Renovating Houses' first aired on Comedy Central in the United States.
  Fact 1.2: 'White People Renovating Houses' first aired on September 13, 2017.
Sentence 2: The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
  Fact 2.0: The twenty-first season of South Park is an American animated sitcom.
  Fact 2.1: The twenty-first season of South Park premiered on Comedy Central on September 13, 2017.
  Fact 2.2: The twenty-first season of South Park will contain ten episodes.

Retrieved items:
<item 1>
Put It Down (South Park)
"Put It Down" is the second episode in the twenty-first season of the American animated television series "South Park".

 The 279th episode of the series overall, it first aired on Comedy Central in the United States on September 20, 2017.
</item 1>

<item 2>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 2>

<item 3>
South Park (season 21)
The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
</item 3>

<item 4>
Holiday Special (South Park)
"Holiday Special" is the third episode in the twenty-first season of the American animated television series "South Park".

 The 280th episode of the series overall, it first aired on Comedy Central in the United States on September 27, 2017.
</item 4>

<item 5>
Chickenlover
"Chickenlover" is the fourth episode in the second season of the American animated television series "South Park".

 The 16th episode of the series overall, it originally aired on Comedy Central in the United States on May 27, 1998.

 The episode was written by series co-creators Trey Parker and Matt Stone, along with David R. Goodman, and directed by Parker.

 In the episode, Officer Barbrady resigns as South Park's only police officer because of his illiteracy.

 Anarchy ensues, just as chickens are mysteriously being molested across South Park.

 Barbrady enlists the help of the boys to learn to read and discover who is molesting the chickens.

 Cartman, meanwhile, masquerades as a police officer.
</item 5>

<item 6>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 6>

<item 7>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 7>

<item 8>
Something Wall-Mart This Way Comes
"Something Wall-Mart This Way Comes" is the ninth episode of the eighth season of the American animated television series "South Park", and the 120th episode of
the series overall.

 It originally aired on Comedy Central in the United States on November 3, 2004.

 In the episode, a Wall-Mart is built in South Park, and the people start to get addicted to shopping from it, due to its irresistibly attractive bargains, thus
leading many businesses in South Park to close down.

 The four boys have to fight against Wall-Mart and to find a way to stop it from taking over the entire town.
</item 8>

<item 9>
White People Renovating Houses
"White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".

 It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
</item 9>

<item 10>
South Park (season 10)
The tenth season of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 22, 2006.

 The tenth season concluded after 14 episodes on November 15, 2006.

 This is the first season in which Kenny does not die and the last season featuring Isaac Hayes (the voice of Chef) as Hayes quit the show following the
backlash behind season nine's "Trapped in the Closet" episode.

 This season also had a minor controversy when the Halloween episode "Hell on Earth 2006" depicted "The Crocodile Hunter's" Steve Irwin with a stingray lodged
in his chest getting thrown out of Satan's Halloween party for not being in costume.

 Episode 2 in this season is the last one with the Braniff Airlines logo.

 All the episodes in this season were written and directed by Trey Parker.
</item 10>
=== END ITEM id=0ddfefcd3fb33835 ===

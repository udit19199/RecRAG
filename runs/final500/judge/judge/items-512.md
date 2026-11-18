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

This file has 10 items.

=== ITEM id=82fba6e9a338768f ===
Gold sentences and facts:
Sentence 0: Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.
  Fact 0.0: Cassiope is a genus of 9 to 12 small shrubby species.
  Fact 0.1: Cassiope is in the family Ericaceae.
Sentence 1: Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to Japan and the Philippines), and Central America and also Europe.
  Fact 1.0: Deutzia is a genus of about 60 species of flowering plants.
  Fact 1.1: Deutzia is in the family Hydrangeaceae.
  Fact 1.2: Deutzia is native to eastern Asia.
  Fact 1.3: Deutzia is native to central Asia.
  Fact 1.4: Deutzia is native from the Himalayas east to Japan and the Philippines.
  Fact 1.5: Deutzia is native to Central America.
  Fact 1.6: Deutzia is also native to Europe.

Retrieved items:
<item 1>
The entries describe diverse plants and one crab genus. Mascarene Dictyosperma album, the wind-shedding “hurricane palm,” is widely cultivated but nearly
extinct in its native range. Canary Islands Adenocarpus viscosus grows above 1,800 m. Dregea vines occur in Africa and southern Asia; Duranta shrubs and trees
are ornamental, with D. erecta invasive in several regions. Tropical Dieffenbachia are popular houseplants
</item 1>

<item 2>
Cassiope lycopodioides
Cassiope lycopodioides, Haida Gwaii mountain-heather or clubmoss mountain heather, is a plant species native to southern Alaska, British Columbia, and the US
State of Washington.

 It is found on rocky slopes in arctic and alpine tundra at elevations up to 2000 m.

 In Washington, it is reported only from King County.

 The specific epithet ""lycopodioides"" refers to the plant's superficial resemblance to some species of clubmoss "(Lycopodium" sensu lato).
</item 2>

<item 3>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 3>

<item 4>
Cassiope mertensiana
Cassiope mertensiana is a species of flowering plant known by the common names western moss heather and white mountain heather.
</item 4>

<item 5>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 5>

<item 6>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 6>

<item 7>
Deutzia silvestrii
Deutzia silvestrii is a plant species in the family Hydrangeaceae.
</item 7>

<item 8>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 8>

<item 9>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 9>

<item 10>
Deutzia ningpoensis
Deutzia ningpoensis is a shrub in the family Hydrangeaceae.

 The species is endemic to China.

 It grows to between 1 and 2.5 metres high and produces panicles of white flowers from May to July in its native range.
</item 10>
=== END ITEM id=82fba6e9a338768f ===

=== ITEM id=76e7a50bad7a35c7 ===
Gold sentences and facts:
Sentence 0: Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic series.
  Fact 0.0: Levni Yilmaz was born in 1973.
  Fact 0.1: Levni Yilmaz is a San Francisco based independent film maker.
  Fact 0.2: Levni Yilmaz is an artist.
  Fact 0.3: Levni Yilmaz is a publisher.
  Fact 0.4: Levni Yilmaz is best known for his 'Tales of Mere Existence' animated comic series.
Sentence 1: Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.
  Fact 1.0: Pamela B. Green is American.
  Fact 1.1: Pamela B. Green is a film producer.
  Fact 1.2: Pamela B. Green is a film director.
  Fact 1.3: Pamela B. Green is known for her work in feature film titles.
  Fact 1.4: Pamela B. Green is known for her work in motion graphics.

Retrieved items:
<item 1>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
</item 1>

<item 2>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 2>

<item 3>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 3>

<item 4>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 4>

<item 5>
Pamela B. Green
Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.

 She is the director and producer of the documentary "Be Natural: The Untold Story of Alice Guy-Blache".
</item 5>

<item 6>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 6>

<item 7>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 7>

<item 8>
Aryanto Yuniawan
Aryanto Yuniawan (born June 18, 1977 in Magelang; age 40) is Chief Executive Officer of PT Mataram Surya Visi Sinema, a multimedia and animation movie company
located in Jogja Indonesia.

 Ary (his nickname) is best known as a director of Battle of Surabaya, the first 2D animation movie in Indonesia which got some awards i.e. Gold Remi Award USA
2016, GrandPrize Winner of SICAF 2016 in South Korea, Winner of Nice International Filmmaker Festival 2017 in France, Winner of Nioda International Film
Festival in India and others.

 He is also acclaimed as a writer, a producer, and currently working on several articles and books related to his job and hobby.

 He started his career from directing and producing independent animation films including music videos.

 He was trusted to be one of the juries of several film festivals such as Festival Film Indonesia (FFI 2016), Noida International Film Festival (NIFF 2016), and
Seoul International Cartoon and Animation Festival (SICAF 2017).
</item 8>

<item 9>
The entries profile filmmakers Pamela B. Green, Ava DuVernay, Howard Barish, Dana Brown, Matthew O’Neill, and the Maya Deren Award; actors Emayatzy Corinealdi
and Issa Rae; musicians Emel Mathlouthi, Siti binti Saad, and Angélique Kidjo; fitness figures Ingrid Romero and Anais Zanotti; and YouTubers Lilly Singh, Wayne
Goss, and Wengie. They
</item 9>

<item 10>
İsmet Ergün
İsmet Ergün (born 1950) is a Berlin artist and stage designer of Turkish provenance.

 Alongside her theatre work she has worked in the world of cinema as an art director and as a production designer.

 The short film which she directed, and for which she herself wrote the screenplay, was singled out for commendation at the 2007 Locarno Film Festival: it also
won the "Best German Film 2007" at the later the same year.
</item 10>
=== END ITEM id=76e7a50bad7a35c7 ===

=== ITEM id=ae60f95742910e44 ===
Gold sentences and facts:
Sentence 0: A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.
  Fact 0.0: A Summer in the Cage is a documentary film.
  Fact 0.1: A Summer in the Cage was released in 2007.
  Fact 0.2: A Summer in the Cage is about a man's experiences with bipolar disorder.
Sentence 1: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.
  Fact 1.0: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary.
  Fact 1.1: American Hardcore: The History of American Punk Rock 1980-1986 was directed by Paul Rachman.
  Fact 1.2: American Hardcore: The History of American Punk Rock 1980-1986 was written by Steven Blush.

Retrieved items:
<item 1>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 1>

<item 2>
Heavy Traffic
Heavy Traffic is a 1973 American adult animated comedy-drama film written and directed by Ralph Bakshi.

 The film, which begins, ends, and occasionally combines with live-action, explores the often surreal fantasies of a young New York cartoonist named Michael
Corleone, using pinball imagery as a metaphor for inner-city life.

 "Heavy Traffic" was Bakshi and producer Steve Krantz's follow-up to the film "Fritz the Cat".

 Though producer Krantz made varied attempts to produce an R-rated film, "Heavy Traffic" was given an X rating by the MPAA.

 The film received positive reviews and is widely considered to be Bakshi's biggest critical success.
</item 2>

<item 3>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 3>

<item 4>
American Pie Presents: Band Camp
American Pie Presents: Band Camp (also known as American Pie: Band Camp) is a 2005 direct-to-DVD sex comedy film.

 It is the first installment in the "American Pie Presents" series and the fourth installment in the "American Pie" franchise.

 It was directed by Steve Rash and written by Brad Riddell.

 The film centers on Matt Stifler, younger brother of Steve Stifler.

 In this film, Matt is sent to the infamous band camp, where he is compelled to change his ways.
</item 4>

<item 5>
Combat Zone Wrestling
Combat Zone Wrestling (CZW) is an American independent wrestling promotion.

 In 1999, John Zandig and five of his students, Ric Blade, T.C.K, Lobo, Nick Gage, and Justice Pain (along with trainer Jon Dahmer) began to run professional
wrestling shows in New Jersey and Delaware, showcasing a brand of hardcore wrestling dubbed as "ultraviolence".

 Ladders, tables, steel folding chairs, thumbtacks, barbed wire, weed whackers, light tubes, panes of glass, and fire are all common elements of "ultraviolent
wrestling" in CZW.

 The company filled a niche for hardcore wrestling fans that had been left open by the folding of Extreme Championship Wrestling (ECW).

 CZW established themselves as the leading American hardcore wrestling promotion at the ECW Arena with their "Cage of Death 3" show in 2001, the year ECW
folded.
</item 5>

<item 6>
Hellfest (American music festival)
Hellfest was an American hardcore punk festival.

 Named for the intense summer heat at the show, the festival originally began featuring hardcore bands before crossing into several genres later on.

 Hellfest was known for its fan-friendly atmosphere, (including a lack of barriers, providing easy access to the stage for stage diving) and for its high
social-awareness, focusing on such causes as animal rights, with many attendees following the straight edge and/or vegan lifestyle.

 It continued as an annual event until it was cancelled in 2005.

 After its cancellation less than thirty-six hours before doors were scheduled to open on the festival's ninth year, some fans, unhappy with the inadequate
processing of refunds, filed small claims court lawsuits against promoter Keith Allen.
</item 6>

<item 7>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 7>

<item 8>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 8>

<item 9>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 9>

<item 10>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 10>
=== END ITEM id=ae60f95742910e44 ===

=== ITEM id=f4e4081f46c86e4c ===
Gold sentences and facts:
Sentence 0: Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All Stars, as well of his own solo band, the Kuwata band.
  Fact 0.0: Keisuke Kuwata's name is written 桑田 佳祐 (Kuwata Keisuke).
  Fact 0.1: Keisuke Kuwata was born February 26, 1956.
  Fact 0.2: Keisuke Kuwata is a Japanese multi-instrumentalist.
  Fact 0.3: Keisuke Kuwata is a Japanese singer-songwriter.
  Fact 0.4: Keisuke Kuwata is frontman for the Southern All Stars.
  Fact 0.5: Keisuke Kuwata has his own solo band, the Kuwata band.
Sentence 1: Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.
  Fact 1.0: Moya Brennan was born Máire Ní Bhraonáin.
  Fact 1.1: Moya Brennan is also known as Máire Brennan.
  Fact 1.2: Moya Brennan was born 4 August 1952.
  Fact 1.3: Moya Brennan is an Irish folk singer.
  Fact 1.4: Moya Brennan is an Irish songwriter.
  Fact 1.5: Moya Brennan is an Irish harpist.
  Fact 1.6: Moya Brennan is an Irish philanthropist.

Retrieved items:
<item 1>
Bruce Yonemoto
Bruce Yonemoto (born 1949) is a Japanese-American multimedia artist.

 His work has been recognized by the National Endowment for the Arts, the American Film Institute, The Rockefeller Foundation, the Maya Deren Award for
Experimental Film and Video, and a mid-career survey show at the Japanese American National Museum, in addition to major solo exhibitions at the
InterCommunication Center in Tokyo, the Institute of Contemporary Art the University of Pennsylvania, and the Kemper Museum in Kansas City, MO.

 Yonemoto was featured in the 2002 Corcoran Gallery Bienniel, in Washington D.C. Yonemoto was also a recipient of the Creative Capital Visual Arts Award in
2008.
</item 1>

<item 2>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 2>

<item 3>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 3>

<item 4>
My Match Is a Makin'
My Match Is A Makin' (English translation of opening track title) is a music album by Irish musicians Moya Brennan and Cormac de Barra.

 This is Moya's eighth studio album to be released.

 It was released on 14 April 2010 exclusively to concert goers on her Spring 2010 tour of the Netherlands.

 The album was re-released to a wider audience under the title Voices & Harps.
</item 4>

<item 5>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 5>

<item 6>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 6>

<item 7>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 7>

<item 8>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 8>

<item 9>
Two Horizons
Two Horizons is the fifth solo album by Irish singer Moya Brennan.

 It is her first full-length release under the name "Moya Brennan", as opposed to "Máire Brennan".

 The album was predominantly recorded in her home studio in Dublin and was nominated for a Grammy award.

 The album was recorded between 2002–2003 and first became available on 23 October 2003.

 It is also Brennan's first non-Christian album since her 1993 recording Misty Eyed Adventures.

 It is the most successful of her solo albums to date and the second to be Grammy-nominated.
</item 9>

<item 10>
Koi Kaze
Koi Kaze (Japanese: 恋風 , lit.

 "Love Wind") is a Japanese manga series written and illustrated by Motoi Yoshida.

 It premiered in the September 2001 issue of "Evening" and ran for 29 chapters until its conclusion in the October 2004 issue.

 The individual chapters were collected and published in five "tankōbon" volumes by Kodansha.

 The series tells of the love affair that develops between 27-year-old Koshiro, and his 15-year-old sister, Nanoka.
</item 10>
=== END ITEM id=f4e4081f46c86e4c ===

=== ITEM id=372ed66ad922b6d4 ===
Gold sentences and facts:
Sentence 0:  Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
  Fact 0.0: Programming is simulcast on WANH (91.5 FM).
  Fact 0.1: WANH (91.5 FM) is located in Meredith.
  Fact 0.2: WANH serves the Lakes Region.
Sentence 1:  The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford County and Merrimack County.
  Fact 1.0: The area comprises all of Belknap County.
  Fact 1.1: The area comprises the southern portion of Carroll County.
  Fact 1.2: The area comprises the eastern portion of Grafton County.
  Fact 1.3: The area comprises the northern portion of Strafford County.
  Fact 1.4: The area comprises the northern portion of Merrimack County.

Retrieved items:
<item 1>
WVNH
WVNH (91.1 FM) is a radio station broadcasting a Christian radio format.

 Licensed to Concord, New Hampshire, United States, the station serves the Concord area.

 The station is currently owned by New Hampshire Gospel Radio, Inc.

 Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
</item 1>

<item 2>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 2>

<item 3>
WLMR
WLMR (1450 AM) is a radio station broadcasting a religious format.

 Licensed to Chattanooga, Tennessee, USA, the station serves the Chattanooga area.

 The station is currently owned by Wilkins Communications Network, Inc. and features programming from USA Radio Network.

 In the early 1980s, the station was automated and played country music from studios in a strip mall on Brainerd Rd.

 It was also Chattanooga's first talk radio station when it held the WZRA call letters.

 WZRA was the first home of Jeff Styles, and also featured well known personalities such as Kelly McCoy and Robert T. Nash.
</item 3>

<item 4>
KWXL-LP
KWXL-LP (98.7 FM) is a high school radio station broadcasting a variety format.

 Licensed to Tucson, Arizona, United States, the station serves the Tucson area.

 The station is currently owned by the Tucson Unified School District.

 KWXL-LP is Tucson's only high school radio station.

 It is also a news radio station for students who attend Pueblo High Magnet School.

 Students broadcasting over the radio station are enrolled in a "Writing/Reporting for Broadcasting" class with instructor Sarah Walson.

 Originally started by Douglas Potter, who retired in 2006.

 Listeners can also find FM 98.7, KWXL on iTunes.
</item 4>

<item 5>
KSCW-DT
KSCW-DT, virtual channel 33 (VHF digital channel 12), is a CW-affiliated television station licensed to Wichita, Kansas, United States.

 The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KSCW and KWCH share studio facilities and KDCU's master control operations are located on East 37th Street North in northeastern Wichita; KSCW maintains
transmitter facilities located in rural northeastern Reno County (due south of Buhler).
</item 5>

<item 6>
KWCH-DT
KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to
Hutchinson.

 The station is owned by Gray Television, as part of a duopoly with CW affiliate KSCW-DT (channel 33); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KWCH and KSCW share studio facilities and KDCU's master control operations are located on 37th Street in northeast Wichita; KWCH maintains transmitter
facilities located in rural northeastern Reno County (south-southeast of Buhler).
</item 6>

<item 7>
KRBT
KRBT (1340 AM) is a radio station broadcasting a sports format.

 Licensed to Eveleth, Minnesota, United States, the station serves the Iron Range area.

 The station is currently owned by Range Broadcasting, Inc., and features programming from the Red Zone Radio Network.

 Originally a stand alone music station with call letters WEVE, then a simulcast music station with WEVE-FM, the station changed it's call letters to KRBT for
"Range's Best Talk" in June of 1998 and discontinued simulcast with WEVE-FM to become a talk station.

 The format change to sports/talk was made following the death of previous owner, Lew Latto, Iron Range Broadcasting, Inc.

 President on August, 24 2011 and subsequent sale to Red Rock Radio in March 2012, and has been retained by the new owner as of July 2017, Range Broadcasting,
Inc.
</item 7>

<item 8>
KVKI-FM
KVKI-FM (96.5 FM) is a radio station broadcasting an adult contemporary music format.

 Licensed to Shreveport, Louisiana, United States, the station serves the Shreveport area.

 The station is currently owned by Townsquare Media.

 Its studios are shared with its other five sister stations in West Shreveport (one mile west of Shreveport Regional Airport), and the transmitter is in
Blanchard, Louisiana.

 The call letters were originally KBCL, and later became religious station KEPT, before the call letters were changed to KVKI to be similar to the call letters
of the popular KVIL radio station in Dallas, Texas.
</item 8>

<item 9>
KKBN
KKBN (93.5 FM) The Cabin a radio station since 1986 broadcasting a Country music format since March 2000.

 Licensed to Twain Harte, California, United States, the station serves the greater Mother Lode area comprising Tuolumne and Calaveras Counties with a combined
population of approximately 110,000.

 The station is currently owned by Clarke Broadcasting Corporation.
</item 9>

<item 10>
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 10>
=== END ITEM id=372ed66ad922b6d4 ===

=== ITEM id=2a99e008d0a5419e ===
Gold sentences and facts:
Sentence 0: The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a unique numeric identifier, referred to as a "DUNS number" to a single business entity.
  Fact 0.0: The Data Universal Numbering System is abbreviated as DUNS or D-U-N-S.
  Fact 0.1: The Data Universal Numbering System is a proprietary system.
  Fact 0.2: The Data Universal Numbering System is developed by Dun & Bradstreet (D&B).
  Fact 0.3: The Data Universal Numbering System is regulated by Dun & Bradstreet (D&B).
  Fact 0.4: The Data Universal Numbering System assigns a unique numeric identifier to a single business entity.
  Fact 0.5: The unique numeric identifier is referred to as a DUNS number.
Sentence 1:  Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
  Fact 1.0: Dun & Bradstreet is often referred to as D&B.
  Fact 1.1: The company's database contains more than 265 million business records.
  Fact 1.2: The company's database contains business records worldwide.

Retrieved items:
<item 1>
Dental notation
Dental professionals, in writing or speech, use several different dental notation systems for associating information with a specific tooth.

 The three most common systems are the ISO System, Universal Numbering System, and Palmer notation method.

 The ISO system is used worldwide, and the Universal is used widely in the United States.

 The ISO System can be easily adapted to computerized charting.
</item 1>

<item 2>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 2>

<item 3>
University of California, Santa Barbara Library
The University of California, Santa Barbara Library is the university library system of the University of California, Santa Barbara in Santa Barbara,
California.

 The Library includes four facilities: Two libraries (the Main Library and the Music Library) and two annexes (Annex I and Annex II).

 The library has some three million print volumes, 30,000 electronic journals, 34,450 e-books, 900,055 digitized items, five million cartographic items
(including some 467,000 maps and 3.2 million satellite and aerial images), more than 3.7 million pieces of microform, 167,500 sound recordings, and 4,100
manuscripts.

 The Library states that it holds 3.2 mi of manuscript and archival collections.

 According to data collected by the American Library Association, as of July 2010 the UC Santa Barbara Library holds the 99th largest library collection in
America, reporting 2,948,999 volumes.
</item 3>

<item 4>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 4>

<item 5>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 5>

<item 6>
Data Universal Numbering System
The Data Universal Numbering System, abbreviated as DUNS or D-U-N-S, is a proprietary system developed and regulated by Dun & Bradstreet (D&B) that assigns a
unique numeric identifier, referred to as a "DUNS number" to a single business entity.

 It was introduced in 1963 to support D&B's credit reporting practice.

 It is standard worldwide.

 DUNS users include the European Commission, the United Nations, and the United States government.

 More than 50 global industry and trade associations recognize, recommend, or require DUNS.

 The DUNS database contains over 250 million entries for businesses throughout the world.
</item 6>

<item 7>
Real estate benchmarking measures and compares property investments; proptech uses technology to improve real estate transactions. Bell Labs is Nokia-owned US
research company. Embase is Elsevier’s extensive biomedical and drug-literature database. Dental notation systems identify teeth; DUNS assigns unique business
identifiers. PIDBA compiles Paleoindian archaeological data. Dun & Bradstreet supplies business data and analytics. TBMA merged into SIFMA in 2006. A periplus
lists coastal ports and distances. BNB
</item 7>

<item 8>
Calypso Ichthyological Database
The Calypso Ichthyological Database numbering system is an open source free repository allowing the unique identification numbering of all fish species with a
six-numeral fixed number.

 This number remains the same throughout any alterations or taxonomic changes to the species' accepted current scientific name and allows for recording of
species data in 32 unique data fields including all previous names in 32 languages.

 It has been in use since 1994 and was pioneered by its inventor, Gerald H. Jennings, with the technical assistance of Terry Hall.

 It is accompanied by a vast photographic library of fish species and is free at the point of use to all researchers and academics.

 Much of the data is also available in published format and online.
</item 8>

<item 9>
Broward County Uniform Station Numbering
The Broward County Uniform Station Numbering system is a system by which all fire stations and fire apparatus in Broward County are numbered.

 The numbering system was developed by the Fire Chief's Association of Broward County in order to minimize confusion and create efficient radio communications
at large fires or emergency scenes, where multiple departments were working together.

 The system went into effect on October 1, 1990 after a large fire at GLS Fiberglass in Port Everglades in 1988.

 Prior to 1990, each fire department had its own numbering system, usually starting at 1 and proceeding to number each station that department operated from.

 At this large fire in Port Everglades, Dania Beach Fire Department, Hollywood Fire Department, Port Everglades Public Safety, and Fort Lauderdale Fire
Department all were on scene with their own Engine 1s.

 As a result, every time a person called for "Engine 1" on fireground radios, confusion would ensue.
</item 9>

<item 10>
Universal Numbering System
The Universal Numbering System is a dental notation system for associating information to a specific tooth, and is commonly used in the United Kingdom.
</item 10>
=== END ITEM id=2a99e008d0a5419e ===

=== ITEM id=621aac07f1d15972 ===
Gold sentences and facts:
Sentence 0: Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic series.
  Fact 0.0: Levni Yilmaz was born in 1973.
  Fact 0.1: Levni Yilmaz is a San Francisco based independent film maker.
  Fact 0.2: Levni Yilmaz is an artist.
  Fact 0.3: Levni Yilmaz is a publisher.
  Fact 0.4: Levni Yilmaz is best known for his 'Tales of Mere Existence' animated comic series.
Sentence 1: Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.
  Fact 1.0: Pamela B. Green is American.
  Fact 1.1: Pamela B. Green is a film producer.
  Fact 1.2: Pamela B. Green is a film director.
  Fact 1.3: Pamela B. Green is known for her work in feature film titles.
  Fact 1.4: Pamela B. Green is known for her work in motion graphics.

Retrieved items:
<item 1>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 1>

<item 2>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 2>

<item 3>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 3>

<item 4>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 4>

<item 5>
Aryanto Yuniawan
Aryanto Yuniawan (born June 18, 1977 in Magelang; age 40) is Chief Executive Officer of PT Mataram Surya Visi Sinema, a multimedia and animation movie company
located in Jogja Indonesia.

 Ary (his nickname) is best known as a director of Battle of Surabaya, the first 2D animation movie in Indonesia which got some awards i.e. Gold Remi Award USA
2016, GrandPrize Winner of SICAF 2016 in South Korea, Winner of Nice International Filmmaker Festival 2017 in France, Winner of Nioda International Film
Festival in India and others.

 He is also acclaimed as a writer, a producer, and currently working on several articles and books related to his job and hobby.

 He started his career from directing and producing independent animation films including music videos.

 He was trusted to be one of the juries of several film festivals such as Festival Film Indonesia (FFI 2016), Noida International Film Festival (NIFF 2016), and
Seoul International Cartoon and Animation Festival (SICAF 2017).
</item 5>

<item 6>
İsmet Ergün
İsmet Ergün (born 1950) is a Berlin artist and stage designer of Turkish provenance.

 Alongside her theatre work she has worked in the world of cinema as an art director and as a production designer.

 The short film which she directed, and for which she herself wrote the screenplay, was singled out for commendation at the 2007 Locarno Film Festival: it also
won the "Best German Film 2007" at the later the same year.
</item 6>

<item 7>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 7>

<item 8>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 8>

<item 9>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
</item 9>

<item 10>
Pamela B. Green
Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.

 She is the director and producer of the documentary "Be Natural: The Untold Story of Alice Guy-Blache".
</item 10>
=== END ITEM id=621aac07f1d15972 ===

=== ITEM id=af98904042180ec3 ===
Gold sentences and facts:
Sentence 0: Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)
  Fact 0.0: Julia Compton Moore's dates are February 10, 1929 to April 18, 2004.
  Fact 0.1: Julia Compton Moore was the wife of Lieutenant General (Ret.).
Sentence 1:  Hal Moore, a United States Army officer.
  Fact 1.0: Hal Moore is a United States Army officer.
Sentence 2: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 2.0: Harold Gregory "Hal" Moore, Jr. lived from February 13, 1922 to February 10, 2017.
  Fact 2.1: Harold Gregory "Hal" Moore, Jr. was a United States Army lieutenant general.
  Fact 2.2: Harold Gregory "Hal" Moore, Jr. was an author.
Sentence 3:  He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point class (1945) to be promoted to brigadier general, major general, and lieutenant general.
  Fact 3.0: Hal Moore was a recipient of the Distinguished Service Cross.
  Fact 3.1: The Distinguished Service Cross is the U.S. military's second highest decoration for valor.
  Fact 3.2: Hal Moore was the first of his West Point class (1945) to be promoted to brigadier general.
  Fact 3.3: Hal Moore was the first of his West Point class (1945) to be promoted to major general.
  Fact 3.4: Hal Moore was the first of his West Point class (1945) to be promoted to lieutenant general.

Retrieved items:
<item 1>
Julia Compton Moore
Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)

 Hal Moore, a United States Army officer.

 Her efforts and complaints in the aftermath of the Battle of Ia Drang prompted the U.S. Army to set up survivor support networks and casualty notification
teams consisting of uniformed officers, which are still in use.
</item 1>

<item 2>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 2>

<item 3>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 3>

<item 4>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 4>

<item 5>
Nadja West
Nadja Y. West (born 1961) is a United States Army lieutenant general and the 44th U.S. Army Surgeon General and Commanding General of the U.S. Army Medical
Command.

 West is the first black Army Surgeon General, and was the first black female, active-duty, major general and the first black female major general in Army
Medicine.

 West is also the first Army black female lieutenant general.

 She is the highest ranking female to have graduated from the United States Military Academy.

 She received the Army Distinguished Service Medal, the Defense Superior Service Medal, the Legion of Merit, and others.
</item 5>

<item 6>
List of current United States first spouses
In the United States, the first spouse (first lady for women, first gentleman for men) is the term used to refer to the spouse of a chief executive - that is,
of the spouse of the President of the United States (the First Lady of the United States and the First Gentleman of the United States) and the spouses of the
governors of the 50 U.S. states and U.S. territories (Guam, Northern Mariana Islands, Puerto Rico, American Samoa, and the United States Virgin Islands) and the
spouse of the mayor of the District of Columbia.

 (The spouses of many mayors are often called the "first lady" or "first gentleman" of the city as well, and the use of the terms sometimes extends even to the
spouses of college presidents).
</item 6>

<item 7>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 7>

<item 8>
Major general (United States)
In the United States Army, United States Marine Corps, and United States Air Force, major general is a two-star general-officer rank, with the pay grade of O-8.

 Major general ranks above brigadier general and below lieutenant general.

 A major general typically commands division-sized units of 10,000 to 15,000 soldiers.

 Major general is equivalent to the two-star rank of rear admiral in the United States Navy and United States Coast Guard and is the highest permanent rank
during peacetime in the uniformed services.

 (The higher ranks are temporary ranks linked to specific positions, although virtually all officers who have been promoted to those ranks are approved to
retire at their highest earned rank.)
</item 8>

<item 9>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 9>

<item 10>
Distinguished Service Cross (United States)
The Distinguished Service Cross is the second highest military award that can be given to a member of the United States Army (and previously, the United States
Army Air Forces and the United States Air Force), for extreme gallantry and risk of life in actual combat with an armed enemy force.

 Actions that merit the Distinguished Service Cross must be of such a high degree that they are above those required for all other U.S. combat decorations but
do not meet the criteria for the Medal of Honor.

 The Distinguished Service Cross is equivalent to the Navy Cross (Navy and Marine Corps), the Air Force Cross (Air Force), and the Coast Guard Cross (Coast
Guard).
</item 10>
=== END ITEM id=af98904042180ec3 ===

=== ITEM id=19a80f178f9b0232 ===
Gold sentences and facts:
Sentence 0:  The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".
  Fact 0.0: The album features a pair of duets with Waylon Jennings.
  Fact 0.1: One of the duets with Waylon Jennings was the "There Ain't No Good Chain Gang" single.
  Fact 0.2: The duet "There Ain't No Good Chain Gang" was one of Johnny Cash's first collaborations with Waylon Jennings.
  Fact 0.3: Johnny Cash and Waylon Jennings recorded songs together throughout the 1980s.
  Fact 0.4: Johnny Cash and Waylon Jennings recorded a separate album entitled "Heroes".
Sentence 1:  Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
  Fact 1.0: Johnny Cash and Waylon Jennings would also work together as The Highwaymen.
  Fact 1.1: The Highwaymen included Willie Nelson.
  Fact 1.2: The Highwaymen included Kris Kristofferson.
Sentence 2: The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
  Fact 2.0: The Highwaymen was a country music supergroup.
  Fact 2.1: The Highwaymen was composed of four of the country music genre's biggest artists.
  Fact 2.2: The four artists were well known for their pioneering influence on the outlaw country subgenre.
  Fact 2.3: The Highwaymen members were Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
Sentence 3:  Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty Records.
  Fact 3.0: The Highwaymen was active as a group between 1985 and 1995.
  Fact 3.1: The Highwaymen recorded three major label albums.
  Fact 3.2: Two of The Highwaymen's major label albums were on Columbia Records.
  Fact 3.3: One of The Highwaymen's major label albums was for Liberty Records.

Retrieved items:
<item 1>
The Highwaymen (country supergroup)
The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country
subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.

 Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty
Records.

 Their Columbia works produced three chart singles, including the number one "Highwayman" in 1985.
</item 1>

<item 2>
Lonesome Pine Fiddlers
The Lonesome Pine Fiddlers (1938- 1966) were an early bluegrass band which included such notable "first generation" bluegrass musicians as Ezra Cline, Bobby
Osborne, Paul Williams, Melvin Goins, Charlie Cline, Curly Ray Cline, Larry Richardson and for a short time Jimmy Martin.

 The group was started by Ezra Cline and Curly Ray Cline and was originally named "Cousin Ezra and the Lonesome Pine Fiddlers".

 The Clines came from a large family consisting of musically talented people.

 Ray and Charlie's father, Charlie, was a talented banjo player and the women in the family, Geraldine and Bobbi, were great singers.

 For reasons unknown, Bobbi and Geraldine never joined the band on the road but often joined in at home, especially when notable Country singers, such as Bill
Monroe, Lester Flatt and Earl Scruggs, and Hank Williams, came visiting.

 None of them ever had a music lesson yet excelled on every instrument they touched.
</item 2>

<item 3>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 3>

<item 4>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 4>

<item 5>
I Would Like to See You Again
I Would Like to See You Again is the 57th album by American country singer Johnny Cash, released on Columbia Records in 1978.

 The title track peaked at #12 on the singles chart, while "There Ain't No Good Chain Gang" reached #2; the album itself peaked at #23.

 The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first
collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".

 Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
</item 5>

<item 6>
Lonesome Pine Fiddlers
In 1954, Charlie did a session, playing lead guitar, with the Stanley Brothers and also another one on RCA with the Fiddlers, although he was not otherwise
working with them at the time.

 He also worked briefly as a sideman with the Osborne Brothers, although he did not record with them.

 By 1958, Charlie (electric lead guitar) and his wife, Lee (electric bass), had rejoined Ezra and Curly Ray in the Lonesome Pine Fiddlers, who were
experimenting with a more modern sound and working a TV show in Huntington, West Virginia, in addition to daily radio in Pikeville.

 In his later years, Charlie was with the Stanley Brothers.

 Curly Ray also played with the Stanley Brothers at a different time as their fiddler.

 Curly Ray was one of the best fiddlers in Bluegrass.

 This most talented family of musicians were the best, surpassed by none.
</item 6>

<item 7>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 7>

<item 8>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 8>

<item 9>
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 9>

<item 10>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 10>
=== END ITEM id=19a80f178f9b0232 ===

=== ITEM id=f6d2ee7adc953316 ===
Gold sentences and facts:
Sentence 0: Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.
  Fact 0.0: Giovanni 'Tinto' Brass was born on 26 March 1933.
  Fact 0.1: Giovanni 'Tinto' Brass is an Italian filmmaker.
Sentence 1: Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.
  Fact 1.0: Baltasar Kormakur Samper was born on 27 February 1966.
  Fact 1.1: Baltasar Kormakur Samper is an Icelandic actor.
  Fact 1.2: Baltasar Kormakur Samper is an Icelandic theater director.
  Fact 1.3: Baltasar Kormakur Samper is an Icelandic film director.
  Fact 1.4: Baltasar Kormakur Samper is an Icelandic film producer.

Retrieved items:
<item 1>
Trasgredire
Trasgredire (Transgressing or Cheeky) is a 2000 sex comedy directed by Tinto Brass, with Yuliya Mayarchuk in the lead role.

 Certain parallelisms are drawn between "Nerosubianco" (1969), another Tinto Brass film set in London.
</item 1>

<item 2>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 2>

<item 3>
Contraband (2012 film)
Contraband is a 2012 American crime thriller film directed by Baltasar Kormákur, starring Mark Wahlberg, Kate Beckinsale, Ben Foster, Caleb Landry Jones,
Giovanni Ribisi, Lukas Haas, Diego Luna and J. K. Simmons.

 The film is a remake of the 2008 Icelandic film "Reykjavík-Rotterdam" which Baltasar Kormákur starred in.

 It was released on January 13, 2012 in the United States by Universal Pictures.
</item 3>

<item 4>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 4>

<item 5>
Baltasar Kormákur
Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.

 He is best known for directing the films "101 Reykjavík", "Hafið", "A Little Trip to Heaven" (starring Julia Stiles and Forest Whitaker), a film based on the
book "Mýrin" "(Jar City)" by Arnaldur Indriðason, "Contraband", "2 Guns" (starring Mark Wahlberg and Denzel Washington) and "Everest".

 His father is the Spanish painter Baltasar Samper and actor Baltasar Breki Samper is his son.
</item 5>

<item 6>
Henrik Dorsin
Per "Henrik" Dorsin (born 20 November 1977) is a Swedish actor, comedian, singer and revue-artist.

 Dorsin started his career as an extra in the 1995 film Vinterviken.

 He started his television career in the satire-show Detta har hänt in 1998.

 He then worked as a revue-artist and dramatist, he produced the revue Slängar av sleven, and has participated in shows with the theater group Stockholms
blodbad.

 He also worked with the SVT entertainment show Säpop, and he has been part of the TV4 comedy show "Parlamentet".

 In 2007 he won the Karamelodiktstipendiet by Povel Ramel.

 In 2014 Dorsin toured with the comedy play "Henrik Dorsin - näktergalen från Holavedsvägen".

 And since 2014 Dorsin runs the theater Scalateatern in Stockholm.
</item 6>

<item 7>
Fermo posta Tinto Brass
Fermo posta Tinto Brass is a 1995 Italian comedy film-erotic film directed by Tinto Brass and set in vignettes.
</item 7>

<item 8>
Baltasar Breki Samper
Baltasar Breki Samper (born 22 July 1989), sometimes referred to as Baltasar Breki Baltasarsson, is an Icelandic actor.

 He is the son of Icelandic director Baltasar Kormákur.

 Before graduating from the Icelandic Academy of the Arts in 2015 Samper helped his father on several films as second assistant director.

 He is known for his role as Hjörtur in the Icelandic TV series "Trapped".
</item 8>

<item 9>
Stjórnin
Stjórnin comprised two singers, Grétar Örvarsson, born July 11, 1959 and Sigríður Beinteinsdóttir, born July 24, 1962.

 Together they represented Iceland at the Eurovision Song Contest 1990 held in Zagreb.

 Their entry, "Eitt lag enn" was placed fourth out of 22 songs.

 Both singers were later members of another Eurovision group Heart 2 Heart who represented Iceland in 1992 with the song "Nei eða já".

 This entry was placed seventh out of 23 in Malmö.
</item 9>

<item 10>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 10>
=== END ITEM id=f6d2ee7adc953316 ===

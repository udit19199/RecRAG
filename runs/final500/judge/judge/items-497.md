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

=== ITEM id=c5775ead7f46d946 ===
Gold sentences and facts:
Sentence 0: Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.
  Fact 0.0: Wangaratta Wahine is an album by the Australian band The Captain Matchbox Whoopee Band.
  Fact 0.1: Wangaratta Wahine was released in 1974.
  Fact 0.2: Wangaratta Wahine was the second album of The Captain Matchbox Whoopee Band.
Sentence 1:  He was declared an Australian Living Treasure by the National Trust of Australia in 1999.
  Fact 1.0: The subject of this sentence was declared an Australian Living Treasure by the National Trust of Australia.
  Fact 1.1: The subject of this sentence received this declaration in 1999.

Retrieved items:
<item 1>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 1>

<item 2>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 2>

<item 3>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 3>

<item 4>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 4>

<item 5>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 5>

<item 6>
Wangaratta Wahine
Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.

 It was recorded at top recording studio Armstrong Studios in Melbourne and the cover art was by famed Australian artist Michael Leunig.
</item 6>

<item 7>
The Preytells
The Preytells were an Australian indie rock band from Perth, formed in 2004 by Jessica Bennett (a.k.a. Audrey Tell) on lead guitar and backing vocals, Simon
Okely (a.k.a. Will Tell) on guitar and lead vocals, Jaclyn Pearson on drums and percussion and Cameron Stewart on bass guitar and backing vocals.

 Their sole album, "Flood Songs/June Songs", was issued in September 2009; before they disbanded later that year.

 Their single, "Shout!"

 ( May 2008), was nominated for WAM Song of the Year in the Pop category in 2008; while, "Lord Hold My Hand", was nominated for the same category in 2009.
</item 7>

<item 8>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 8>

<item 9>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 9>

<item 10>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 10>
=== END ITEM id=c5775ead7f46d946 ===

=== ITEM id=02e0f5498dc646e1 ===
Gold sentences and facts:
Sentence 0: "Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.
  Fact 0.0: 'Love Incredible' is a song by Norwegian DJ and record producer Cashmere Cat.
  Fact 0.1: 'Love Incredible' features vocals by Cuban-American singer Camila Cabello.
Sentence 1:  It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
  Fact 1.0: 'Love Incredible' was released on 17 February 2017.
  Fact 1.1: 'Love Incredible' was released through Interscope and Mad Love.
  Fact 1.2: 'Love Incredible' was the third single from the debut album '9' of Cashmere Cat.
Sentence 2:  An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
  Fact 2.0: An imprint of Interscope Geffen A&M Records has Universal Music Group as its parent company.
  Fact 2.1: Universal Music Group is a subsidiary of Vivendi S.A.

Retrieved items:
<item 1>
Incredible Connection
Incredible Connection (Pty) Ltd is the largest consumer electronics and IT retailer in South Africa.

 On 15 December 1998, Incredible Connection was acquired by Connection Group Holdings Limited and now operates as subsidiary of that company, itself a
subsidiary of JD Group Limited.

 As of 2005, Incredible Connection has 34 retail outlets in South Africa, and in 2002 it was reported to hold 45% of the IT retail market in the country.
</item 1>

<item 2>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 2>

<item 3>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 3>

<item 4>
Nobody Else (CeCe Peniston song)
"Nobody Else" is a 1998 song by CeCe Peniston.

 The composition was to be the lead single of the singer's originally scheduled album release on the Silk Entertainment label, as well as reportedly released on
Steve Hurley's compilation "The Voices of Life, Vol 1."

 The production of her album was cancelled eventually, and releasing of the compilation wouldn't happen either.

 However, the singer released another two singles on the label, "He Loves Me 2" the following year and "My Boo" in 2000.
</item 4>

<item 5>
No Promises (Icehouse song)
"No Promises" is the first single released by Australian band, Icehouse from the band's 1986 album, "Measure for Measure" and was released in November 1985 on
Regular Records as 7" Vinyl Single and 12" Vinyl Single formats.

 It was also released in the UK and Europe by Chrysalis Records again on 7" and 12" formats but with different track listings.

 It was subsequently released in the US by Chrysalis on 7" and 12" formats, again with different track listings.

 It peaked at #30 on the Australian singles charts.

 A remix version by [love] tatto was released on the Icehouse album "Meltdown" in 2002.
</item 5>

<item 6>
Kinetic Records
The label was distributed by Reprise/Warner Bros.

 Records from its inception until 2001, when Kinetic switched distribution to BMG.

 The latter eventually absorbed the label during the merger with Sony Music.
</item 6>

<item 7>
Breathing Your Love
"Breathing Your Love" is a song by the Swedish singer Darin featuring vocals by singer Kat DeLuna and the first single from "Flashback".

 The song co-written by Darin with RedOne, Bilal Hajji and Novel was released to radio stations and as a digital download in Sweden on 8 October 2008.

 The single was released in Finland and is also the first single from Darin to be released in the United Kingdom in January 2010.

 In October 2009 Swedish astronaut Christer Fuglesang took 2 copies of the single into space with him, as his daughter is a Darin fan, a photo of the disc can
be seen with earth in the distance can be found on the official Darin website.
</item 7>

<item 8>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 8>

<item 9>
Interscope Records
Interscope Records is an American major record label.

 An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
</item 9>

<item 10>
Love Incredible
"Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.

 It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
</item 10>
=== END ITEM id=02e0f5498dc646e1 ===

=== ITEM id=46aa679a0c519da3 ===
Gold sentences and facts:
Sentence 0: Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a 1760 play by Francesco Ringhieri, "L'Osiride".
  Fact 0.0: Mosè in Egitto (Moses in Egypt) is a three-act opera.
  Fact 0.1: Mosè in Egitto was written by Gioachino Rossini.
  Fact 0.2: Mosè in Egitto has an Italian libretto by Andrea Leone Tottola.
  Fact 0.3: The libretto of Mosè in Egitto was based on a 1760 play by Francesco Ringhieri.
  Fact 0.4: The 1760 play by Francesco Ringhieri is titled L'Osiride.
Sentence 1: Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.
  Fact 1.0: Lucrezia Borgia is a melodramatic opera.
  Fact 1.1: Lucrezia Borgia is in a prologue and two acts.
  Fact 1.2: Lucrezia Borgia is by Gaetano Donizetti.

Retrieved items:
<item 1>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 1>

<item 2>
Lucrezia Borgia (1912 film)
Lucrezia Borgia is a 1912 Italian silent historical film directed by Gerolamo Lo Savio and starring Francesca Bertini in the title role of Lucrezia Borgia.
</item 2>

<item 3>
Lucrezia Borgia (1940 film)
Lucrezia Borgia is a 1940 Italian historical film directed by Hans Hinrich and starring Isa Pola, Friedrich Benfer and Carlo Ninchi.

 The film portrays the life of Lucrezia Borgia (1480-1519), one of a number of Italian films of the era set during the Renaissance.

 It was made at the Scalera Studios in Rome.
</item 3>

<item 4>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 4>

<item 5>
Lucrezia Borgia (1922 film)
Lucrezia Borgia is a 1922 German silent historical film directed by Richard Oswald and starring Conrad Veidt, Liane Haid and Albert Bassermann.

 It was based on a novel by Harry Sheff, and portrayed the life of the Renaissance Italian aristocrat Lucrezia Borgia (1480-1519).

 Botho Hoefer and Robert Neppach worked as the film's art directors, designing the period sets needed.
</item 5>

<item 6>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 6>

<item 7>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 7>

<item 8>
Mosè in Egitto
Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a
1760 play by Francesco Ringhieri, "L'Osiride".

 It premièred on 5 March 1818 at the recently reconstructed Teatro San Carlo in Naples, Italy.
</item 8>

<item 9>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 9>

<item 10>
Lucrezia Borgia (1947 film)
Lucrezia Borgia (Spanish:Lucrecia Borgia) is a 1947 Argentine historical film directed by Luis Bayón Herrera and starring Olinda Bozán, Héctor Quintanilla and
Gogó Andreu.

 The film portrays the life of Lucrezia Borgia (1480-1519).
</item 10>
=== END ITEM id=46aa679a0c519da3 ===

=== ITEM id=0abb5ddd46b49d1d ===
Gold sentences and facts:
Sentence 0: Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.
  Fact 0.0: Disteghil Sar is also called Distaghil Sar.
  Fact 0.1: Disteghil Sar is the highest mountain in the Hispar Muztagh.
  Fact 0.2: The Hispar Muztagh is a subrange of the Karakoram mountain range.
  Fact 0.3: Disteghil Sar is in Gilgit-Baltistan.
Sentence 1:  The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
  Fact 1.0: Disteghil Sar has a top ridge about 3 km long.
  Fact 1.1: The top ridge of Disteghil Sar is above 7400 m.
  Fact 1.2: Disteghil Sar has three distinct summits.
  Fact 1.3: The (north)west summit of Disteghil Sar is 7885 m high.
  Fact 1.4: The central summit of Disteghil Sar is 7760 m high.
  Fact 1.5: The (south)east summit of Disteghil Sar is 7696 m or 7535 m high.
Sentence 2:  Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
  Fact 2.0: The summit of Disteghil Sar has an elevation of 7295 m.
  Fact 2.1: Disteghil Sar is the highest peak in the Yengisogat subrange of the Karakoram.

Retrieved items:
<item 1>
The Crown (mountain)
The Crown, also known as Huang Guan Shan and sometimes Crown Peak, is a mountain in the Karakoram mountain range in China.

 It is located in the Xinjiang Uyghur Autonomous Region of China.

 Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
</item 1>

<item 2>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 2>

<item 3>
Kezhen Peak
Kezhen Peak, also known as Karpogo Sar, is a mountain in the Karakoram mountain range.

 It is located in the Xinjiang Autonomous Region of China.
</item 3>

<item 4>
Kanjut Sar
Kanjut Sar (Urdu: کنجت سر‎ ) or Kunjudh Sar as pronounced in "Wakhi" is a mountain located in the Hispar Muztagh, a subrange of the Karakoram mountain range.

 Kunjudh Sar in wakhi language mean that which overlooks Kunjudh, or above Kunjudh, while Khujudh is the wakhi name for Lower Hunza.

 It is the 26th highest mountain on Earth and the 11th highest in Pakistan.
</item 4>

<item 5>
Passu Sar
Passu Sar (Urdu: ‎ ; or Passu Sar, Passu I) is a mountain peak in the Batura Muztagh, a sub-range of the Karakoram mountain range, located in the Gilgit
District of Gilgit-Baltistan, Pakistan, west of the Hunza Valley.

 It is the high point of the Passu massif, which also includes Passu Diar (or "Passu East", "Pasu II").

 The peak lies on the main ridge of the Batura Muztagh, about 7 km (4 mi) east of Batura Sar.
</item 5>

<item 6>
Yengisogat
The Yengisogat range (音苏盖提 ), also known as the Wesm Mountains, is a Chinese subrange of the Karakoram mountain range.

 It lies north of the Baltoro Muztagh, home of the eight-thousanders of the Karakoram.

 The highest peak is Huangguan Shan, or Crown Peak, 7,265 m (23,835 ft) (also sometimes given as 7,295 m/23,934 ft).
</item 6>

<item 7>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 7>

<item 8>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 8>

<item 9>
Yutmaru Sar
Yutmaru Sar is a mountain in the Hispar mountain range, a subrange of the Karakoram.

 At an elevation of 7283 m it is the 88th highest mountain in the world.

 Yutmaru Sar is located in Gilgit-Baltistan, Pakistan.

 It was first climbed in 1980.
</item 9>

<item 10>
Momhil Sar
Momhil Sar, or Mumhail Sar as pronounced in Wakhi is at 7343 m above sea level, is the 64th highest mountain peak in the world.

 Mumhail Sar in Wakhi means the mountain that overlooks or is above Grandmother's cattle pen or paddock.

 It is situated in the Hispar Muztagh subrange of the Karakoram range, a few kilometres to the north-west of its parent peak Trivor.
</item 10>
=== END ITEM id=0abb5ddd46b49d1d ===

=== ITEM id=fa2465efefbbdab7 ===
Gold sentences and facts:
Sentence 0: Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.
  Fact 0.0: Ronkonkoma is a major railroad station.
  Fact 0.1: Ronkonkoma is a major transportation hub.
  Fact 0.2: Ronkonkoma is along the Main Line of the Long Island Rail Road.
  Fact 0.3: The Ronkonkoma railroad station is in Ronkonkoma, New York.
Sentence 1: Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.
  Fact 1.0: Ronkonkoma is a hamlet.
  Fact 1.1: Ronkonkoma is a census-designated place (CDP).
  Fact 1.2: Ronkonkoma is on Long Island.
  Fact 1.3: Ronkonkoma is in the Town of Islip.
  Fact 1.4: The Town of Islip is in New York, United States.

Retrieved items:
<item 1>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 1>

<item 2>
Ronkonkoma (LIRR station)
Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.

 The station is the eastern terminus of the Ronkonkoma Branch and the western terminus of the out Branch.
</item 2>

<item 3>
Great Neck, NY, encompasses nine North Hempstead villages/hamlets (~40,000 residents), five ZIP codes and one school district; Saddle Rock is its Blue Ribbon
K–5 school. Floral Park village had 15,863 residents in 2010. Old Country Road serves central Nassau shopping and partly borders Hempstead/North Hempstead. Long
Island’s 75-mile outer barrier separates lagoons from the Atlantic. The Nassau Herald serves Five Towns. Other entries describe
</item 3>

<item 4>
Ridgefield Park station
Ridgefield Park Station, also known as West Shore Station, was railroad station in Ridgefield Park, New Jersey at the foot of Mount Vernon Street served by the
New York, Susquehanna and Western Railroad (NYSW) and the West Shore Railroad, a division of New York Central (NYCRR) The New York, Ontario and Western Railway
(NYO&W) had running rights along the West Shore and sometimes stopped at Ridgefield Park.

 First opened in 1883 it was one of three passenger stations in the village, the others being the Little Ferry Station to the south and Westview Station to the
north.

 The station house, built at a cost $100,000 opened in 1927.

 Southbound service crossed Overpeck Creek and continued to terminals on the Hudson River waterfront where there was connecting ferry service across the Hudson
River to Manhattan.

 Northbound near Bogota the parallel NYSW and West Shore lines diverge and continue into northern New Jersey, Pennsylviania, and upstate New York.

 Passenger service ended in 1966.
</item 4>

<item 5>
Lacona Railroad Station and Depot
Lacona Railroad Station and Depot, also known as New York Central and Hudson River Railroad Station and Depot, is a historic railway depot located at Lacona in
Oswego County, New York.

 It was built in 1891 by the New York Central and Hudson River Railroad.

 It is a small rectangular, one story, gable ended structure.
</item 5>

<item 6>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 6>

<item 7>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 7>

<item 8>
North Creek Railroad Station
The North Creek Railroad Station is a historic railroad station complex located at North Creek, Warren County, New York.

 The complex consists of the railroad station, the freight house, round house, turntable, and horse barn.

 The station was built in 1874 and is a simple, rectangular, gable roofed building with a broad, overhanging strut-supported roof in the Stick-Eastlake style.

 Its exterior is covered with vertical boards.
</item 8>

<item 9>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 9>

<item 10>
Long Island Electric Railway
The Long Island Electric Railway was a streetcar company operating in Brooklyn, Queens, and Nassau County, New York, United States between 1894 and 1926.

 The company was partially owned by the Long Island Consolidated Electric Companies, a holding company for the Long Island Rail Road and partially by August
Belmont and the Interborough Rapid Transit Company.

 It connected the east end of the Fulton Street El at Crescent Street station in City Line, Brooklyn with Jamaica, Queens, and ran from there to the Nassau
County line at Queens Village and to Far Rockaway, Queens via Nassau County.

 It also had a connection to Belmont Park.

 The New York and Long Island Traction Company used trackage rights over its line from Crescent Street to Queens Village.
</item 10>
=== END ITEM id=fa2465efefbbdab7 ===

=== ITEM id=5b659feb094959b0 ===
Gold sentences and facts:
Sentence 0:  Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
  Fact 0.0: Dan Hurley is the son of Bob Hurley.
  Fact 0.1: Bob Hurley is a Hall of Fame high school coach.
  Fact 0.2: Dan Hurley is the younger brother of Bobby Hurley.
  Fact 0.3: Bobby Hurley is a former Duke guard.
  Fact 0.4: Bobby Hurley is a former Sacramento Kings guard.
Sentence 1: Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.
  Fact 1.0: Robert Matthew Hurley was born on June 28, 1971.
  Fact 1.1: Robert Matthew Hurley is an American basketball coach.
  Fact 1.2: Robert Matthew Hurley is a former college player.
  Fact 1.3: Robert Matthew Hurley is a former professional player.
Sentence 2:  Hurley is currently the head coach of the Arizona State men's team.
  Fact 2.0: Hurley is currently the head coach of the Arizona State men's team.

Retrieved items:
<item 1>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 1>

<item 2>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 2>

<item 3>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 3>

<item 4>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 4>

<item 5>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 5>

<item 6>
Rhode Island Rams men's basketball
The Rhode Island Rams men's basketball team is a college basketball program that competes in NCAA Division I and the Atlantic 10 Conference representing the
University of Rhode Island.

 The team is under the direction of head coach Dan Hurley, who was hired on March 20, 2012.

 The Rams play their home games at the Ryan Center (capacity 7,657).
</item 6>

<item 7>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 7>

<item 8>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 8>

<item 9>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 9>

<item 10>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 10>
=== END ITEM id=5b659feb094959b0 ===

=== ITEM id=11875c9559fa24a0 ===
Gold sentences and facts:
Sentence 0:  Related fields of study include Islamic philosophy and Sufism.
  Fact 0.0: Related fields of study include Islamic philosophy.
  Fact 0.1: Related fields of study include Sufism.
Sentence 1: Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular] values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most important and central crystallization of" mystical practice in Islam.
  Fact 1.0: Sufism is also called Taṣawwuf.
  Fact 1.1: The Arabic name for Sufism is الْتَّصَوُّف [sect].
  Fact 1.2: The personal noun for a Sufi is صُوفِيّ, transliterated as 'ṣūfiyy' or 'ṣūfī'.
  Fact 1.3: The personal noun mutaṣawwūf is مُتَصَوّف.
  Fact 1.4: Sufism is often defined as 'Islamic mysticism'.
  Fact 1.5: Sufism is often defined as 'the inward dimension of Islam'.
  Fact 1.6: Sufism is often defined as 'the phenomenon of mysticism within Islam'.
  Fact 1.7: Sufism is a mystical trend in Islam.
  Fact 1.8: Sufism is characterized by particular values, ritual practices, doctrines and institutions.
  Fact 1.9: Sufism began very early on in Islamic history.
  Fact 1.10: Sufism represents the main manifestation of mystical practice in Islam.
  Fact 1.11: Sufism represents the most important and central crystallization of mystical practice in Islam.

Retrieved items:
<item 1>
Moinuddin Chishti
Chishtī Muʿīn al-Dīn Ḥasan Sijzī (1142–1236 CE), known more commonly as Muʿīn al-Dīn Chishtī or Moinuddin Chishti, or reverently as a Shaykh Muʿīn al-Dīn or
Hazrat Muʿīn al-Dīn or Khwājā Muʿīn al-Dīn by South Asian Muslims, was a Persian Muslim preacher, ascetic, religious scholar, philosopher, and mystic from
Sistan, who eventually ended up settling in the Indian subcontinent in the early 13th-century, where he promulgated the famous Chishtiyya order of Sunni
mysticism.

 This particular "tariqa" (order) became the dominant Muslim spiritual group in medieval India and many of the most beloved and venerated Indian Sunni saints
were Chishti in their affiliation, including Nizamuddin Awliya (d. 1325) and Amir Khusrow (d. 1325).
</item 1>

<item 2>
A Hidden Treasure
Hidden Treasure (Arabic: کنزاً مخفیاً‎ ‎ ) is a Hadith Qudsi that has a very prominent role in Islamic mysticism and Islamic philosophy.
</item 2>

<item 3>
Sufism
Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic
mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular]
values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most
important and central crystallization of" mystical practice in Islam.

 Although the overwhelming majority of Sufis, both pre-modern and modern, have been adherents of Sunni Islam, there nevertheless also developed certain strands
of Sufi practice within the ambit of Shia Islam during the late medieval period.
</item 3>

<item 4>
Ovamir Anjum
Ovamir Anjum is the Imam Khattab Chair of Islamic Studies at the Department of Philosophy, University of Toledo.

 He studies the connections between theology, ethics, politics, and law in classical and medieval Islam, with a subfocus on its comparisons with western
thought.

 Related fields of study include Islamic philosophy and Sufism.
</item 4>

<item 5>
Omid Safi
Omid Safi is an American Professor of Asian and Middle Eastern Studies at Duke University, where he is the Director of Duke Islamic Studies Center, and a
columnist for On Being.

 Dr. Safi specializes in Islamic mysticism (Sufism), contemporary Islamic thought and medieval Islamic history.

 He has served on the board of the Pluralism project at Harvard University and is the co-chair of the steering committee for the Study of Islam and the Islamic
Mysticism Group at the American Academy of Religion.

 Before joining Duke University, Dr. Safi was a professor at the University of North Carolina Chapel Hill.
</item 5>

<item 6>
Moinuddin Chishti
1325) and Amir Khusrow (d. 1325).

 As such, Muʿīn al-Dīn Chishtī's legacy rests primarily on his having been "one of the most outstanding figures in the annals of Islamic mysticism."

 Additionally, Muʿīn al-Dīn Chishtī is also notable for having been one of the first major Islamic mystics to formally allow his followers to incorporate the
"use of music" in their devotions, liturgies, and hymns to God, which he did so in order to make the foreign Arab faith more relatable to the indigenous peoples
who had recently entered the religion or whom he sought to convert.
</item 6>

<item 7>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 7>

<item 8>
Irfan
In Islam, ‘Irfaan (Arabic/Persian/Urdu: عرفان ; Turkish: "İrfan" ), also spelt "Irfaan" and "Erfan", literally ‘knowledge, awareness, wisdom’, is gnosis.

 Islamic mysticism can be considered as a vast range that engulfs theoretical and practical and conventional mysticism and has been intertwined with sufism and
in some cases they are assumed identical.

 however Islamic mysticism is assumed as one of the Islamic sciences alongside theology and philosophy.

 Islamic’s mysticism is cognition and knowledge that love has been intertwined through it with structure of revelation in Islam.
</item 8>

<item 9>
Dawud Tai
Abu Solaiman Dawud ibn Nosair al-Ta’i, usually referred to as Dawud Tai, (died between 777 and 782) was a Islamic scholar and Sufi mystic.

 His disciples included many influential personalities of Islamic mysticism, e.g., Maruf Karkhi and Ahmad al-Buni.
</item 9>

<item 10>
Reynold A. Nicholson
Reynold Alleyne Nicholson, or R. A. Nicholson (18 August 1868 – 27 August 1945), was an eminent English orientalist, scholar of both Islamic literature and
Islamic mysticism and widely regarded as one of the greatest Rumi (Mevlana or Mawlana) scholars and translators in the English language.
</item 10>
=== END ITEM id=11875c9559fa24a0 ===

=== ITEM id=3e43aa87c2825e08 ===
Gold sentences and facts:
Sentence 0: Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
  Fact 0.0: Hänt Extra is a Swedish language magazine.
  Fact 0.1: Hänt Extra is a weekly magazine.
  Fact 0.2: Hänt Extra is a celebrity and youth magazine.
  Fact 0.3: Hänt Extra is published by Aller Media.
  Fact 0.4: Hänt Extra is published in Stockholm, Sweden.
Sentence 1: Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
  Fact 1.0: Tea Moderna is a Macedonian woman's weekly magazine.
  Fact 1.1: Tea Moderna's name in Macedonian is Теа Модерна.
  Fact 1.2: Tea Moderna is a magazine for women.

Retrieved items:
<item 1>
Katarina Ivanovska
Katarina Ivanovska (Macedonian: Катарина Ивановска ; born 18 August 1988) is a Macedonian model and actress.

 She began her modeling career in 2004, appearing at Milan Fashion Week after winning the Look Models International model search in Macedonia.

 In December, 2004, she appeared in a pictorial for "Elle" magazine and has also appeared in "Citizen K", "Stiletto" and the Italian and Russian "Vogue".

 She has been featured on the covers of "Diva" and "Maxima" magazines and in advertisements for D&G in 2006.

 She is considered the most successful Macedonian model.

 In 2010, Ivanovska appeared in Serbian "Elle" magazine.

 In 2011 she signed a contract for advertising Victoria's Secret products.

 In 2011 she got her first acting job in the Macedonian World War II film, "The Third Half", landing the lead role of a young Jewish girl named Rebecca.
</item 1>

<item 2>
Tennis (magazine)
TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.

 There are eight issues per year covering the world of tennis.

 The "tennis.com" website is an offshoot of the magazine and part of the same media company.
</item 2>

<item 3>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 3>

<item 4>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 4>

<item 5>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 5>

<item 6>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 6>

<item 7>
Nikola Mladenov
Nikola Mladenov (Macedonian: Никола Младенов ; 10 March 1964 – 26 March 2013) was a Macedonian journalist, actor, founder and managing editor of the weekly
newspaper "Fokus", widely regarded as one of the most reputable journalists in Macedonia, who along with his colleagues from the newspaper "Mlad borec" is
considered to have laid the basis for the pluralistic principles in the Macedonian journalism.

 The "Award for best research story of the year – Nikola Mladenov", which is awarded by the Macedonian Institute for Media, is named after him.
</item 7>

<item 8>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 8>

<item 9>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 9>

<item 10>
2015 Macedonian Football Supercup
The 2015 Macedonian Supercup was the 3rd Macedonian Football Supercup, an annual Macedonian football match played between the winners of the previous season's
First League and Macedonian Cup.

 The game will be played between Rabotnički, who beat Teteks to win the 2015 Macedonian Cup Final, and Vardar, champions of the 2014–15 First League.

 It was played at Philip II Arena, Skopje and was won by Vardar by 4–3 on penalties after the match was ended 1–1 after the regular and extra time.
</item 10>
=== END ITEM id=3e43aa87c2825e08 ===

=== ITEM id=473d37b3cb869d0d ===
Gold sentences and facts:
Sentence 0: Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
  Fact 0.0: Miltoniopsis is a genus of orchids.
  Fact 0.1: Miltoniopsis is abbreviated Mltnps in horticultural trade.
  Fact 0.2: Miltoniopsis is native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
Sentence 1: Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few species extending north into temperate regions of Asia.
  Fact 1.0: Strobilanthes is a genus of about 350 species of flowering plants.
  Fact 1.1: Strobilanthes is in the family Acanthaceae.
  Fact 1.2: Strobilanthes is mostly native to tropical Asia and Madagascar.
  Fact 1.3: A few Strobilanthes species extend north into temperate regions of Asia.

Retrieved items:
<item 1>
Miltoniopsis vexillaria
Miltoniopsis vexillaria ("the flag-like "Miltoniopsis"") is a species of epiphytic orchid in the genus Miltoniopsis.
</item 1>

<item 2>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 2>

<item 3>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 3>

<item 4>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 4>

<item 5>
Broughtonia
Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.

 The genus is abbreviated Bro in trade journals.
</item 5>

<item 6>
Miltoniopsis
Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.

 This genus comprises 5 species.

 Miltoniopsis's common name is Pansy Orchid.
</item 6>

<item 7>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 7>

<item 8>
Hemigraphis
Hemigraphis is a genus of plants in the family Acanthaceae, consisting of about 30 species native to tropical Asia.

 "Hemigraphis" is sometimes included in the genus "Strobilanthes"
</item 8>

<item 9>
Strobilanthes japonica
Strobilanthes japonica is a flowering herbaceous perennial plant from Asia, one of around 350 plants of the genus "Strobilanthes".

 The 20–50 cm ornamental plant is cultivated in Japan and China, and blooms in autumn with 1.5 cm purple to white funnel-shaped flowers.
</item 9>

<item 10>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 10>
=== END ITEM id=473d37b3cb869d0d ===

=== ITEM id=bdd37407bd672a18 ===
Gold sentences and facts:
Sentence 0: The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista Distribution.
  Fact 0.0: The Rescuers is a 1977 film.
  Fact 0.1: The Rescuers is an American animated film.
  Fact 0.2: The Rescuers is a comedy-drama adventure film.
  Fact 0.3: The Rescuers was produced by Walt Disney Productions.
  Fact 0.4: The Rescuers was first released on June 22, 1977.
  Fact 0.5: The Rescuers was first released by Buena Vista Distribution.
Sentence 1: The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.
  Fact 1.0: The Muppets is a 2011 film.
  Fact 1.1: The Muppets is an American film.
  Fact 1.2: The Muppets is a musical comedy film.
  Fact 1.3: The Muppets is the seventh theatrical film featuring the Muppets.

Retrieved items:
<item 1>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 1>

<item 2>
The Muppets at Walt Disney World
The Muppets at Walt Disney World is a television special starring Jim Henson's Muppets at Walt Disney World in Lake Buena Vista, Florida.

 The special aired on NBC as part of "The Magical World of Disney" on May 6, 1990, and turned out to be the last Muppets special that Henson would work on (he
would die 10 days following its airing, on May 16).
</item 2>

<item 3>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 3>

<item 4>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 4>

<item 5>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 5>

<item 6>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 6>

<item 7>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 7>

<item 8>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 8>

<item 9>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 9>

<item 10>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 10>
=== END ITEM id=bdd37407bd672a18 ===

=== ITEM id=79b6a08b4fc97ca0 ===
Gold sentences and facts:
Sentence 0: Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.
  Fact 0.0: Edward Benjamin Britten, Baron Britten, lived from 22 November 1913 to 4 December 1976.
  Fact 0.1: Edward Benjamin Britten, Baron Britten, was an English composer.
  Fact 0.2: Edward Benjamin Britten, Baron Britten, was an English conductor.
  Fact 0.3: Edward Benjamin Britten, Baron Britten, was an English pianist.
Sentence 1: Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.
  Fact 1.0: Béla Viktor János Bartók lived from 25 March 1881 to 26 September 1945.
  Fact 1.1: Béla Viktor János Bartók was a Hungarian composer.
  Fact 1.2: Béla Viktor János Bartók was a Hungarian pianist.
  Fact 1.3: Béla Viktor János Bartók was a Hungarian ethnomusicologist.

Retrieved items:
<item 1>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 1>

<item 2>
List of string quartets by Béla Bartók
The Hungarian composer Béla Bartók wrote six string quartets, each for the usual forces of two violins, viola and cello.

 Notable composers who have been influenced by them include Benjamin Britten, particularly in the Sonata in C for Cello and Piano (; ), Elliott Carter, who
refers in the opening of his own First String Quartet to Bartók’s Sixth Quartet , Chen Yi , Edison Denisov, whose Second Quartet is closely related to Bartók’s
Fifth Quartet , Franco Donatoni, who was deeply impressed when he heard a broadcast of Bartók's Fourth Quartet , Robert Fripp, who mentions them as an influence
upon King Crimson , Miloslav Ištvan , György Kurtág, whose Opp.
</item 2>

<item 3>
List of string quartets by Béla Bartók
1 and 28 both owe a great deal to Bartók's quartets (; ), György Ligeti, whose two string quartets both owe a great deal to Bartók’s quartets (; ), Bruno
Maderna , George Perle, who credits the Bartók Fourth and Fifth Quartets as precedents for his use of arrays of chords related to one another by different types
of symmetry , Walter Piston (; ), Kim Dzmitrïyevich Tsesakow , Wilfried Westerlinck , Stefan Wolpe, who explained in a public lecture how he had derived ideas
from Bartók’s Fourth Quartet , and Xu Yongsan .
</item 3>

<item 4>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 4>

<item 5>
Béla Bartók Music High School
Béla Bartók Music High School (Bartók Béla Zeneművészeti Szakközépiskola) is situated in the Palace of Music (Zenepalota) in Bartók square Miskolc, Hungary.

It is a famous Music school named after the famous Hungarian composer Béla Bartók.

It was founded in 1966.
</item 5>

<item 6>
Debussy’s *Masques* (1904), premiered by Ricardo Viñes in 1905, evokes existential tragedy amid his marital separation. Elizabeth Barrett Browning’s 44 *Sonnets
from the Portuguese* (written 1845–46; published 1850) remain popular. Bartók (1881–1945), Hungarian composer and ethnomusicology pioneer, wrote six influential
quartets. His Sixth, begun in Switzerland in 1939 and completed
</item 6>

<item 7>
String Quartet No. 6 (Bartók)
The String Quartet No. 6, Sz.

 114, BB 119, was the final string quartet that Béla Bartók wrote before his death.

 It was begun in August 1939 in Saanen, Switzerland, where Bartók was a guest of his patron, the conductor Paul Sacher.

 Shortly after he completed the Divertimento for String Orchestra on the 17th, he started on a commission for his friend, the violinist Zoltán Székely.

 Székely was acting as intermediary for the "New Hungarian Quartet", who had given the Budapest premiere of the String Quartet No. 5.

 With the outbreak of World War II and his mother's illness, Bartók returned to Budapest, where the quartet was finished in November.

 After his mother's death, Bartók decided to leave with his family for the United States.
</item 7>

<item 8>
Suite, Op. 14 (Bartók)
The Suite, Op. 14, Sz.

 62, BB 70 is a piece for solo piano written by Béla Bartók.

 It was written in February 1916, published in 1918, and debuted by the composer on April 21, 1919, in Budapest.

 The Suite is one of Bartók's most significant works for piano, only comparable with his 1926 Piano Sonata.

 Though much of Bartók's work makes frequent use of Eastern European folk music, this suite is one of the few pieces without melodies of folk origin.

 However, Romanian, Arabic, and North African rhythmic influences can still be found in some movements.

 Originally intending the suite to be a five-movement work, Bartók later decided against the idea and discarded the second movement, the "Andante", which was
published only posthumously in the October 1955 issue of "Új Zenei Szemle" (New Musical Review).
</item 8>

<item 9>
Benjamin Britten
Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.

 He was a central figure of 20th-century British classical music, with a range of works including opera, other vocal music, orchestral and chamber pieces.

 His best-known works include the opera "Peter Grimes" (1945), the "War Requiem" (1962) and the orchestral showpiece "The Young Person's Guide to the Orchestra"
(1945).
</item 9>

<item 10>
List of compositions by Béla Bartók
This aspires to be a complete list of compositions by Béla Bartók.

 The catalogue numbering by András Szőllősy (Sz.)

, László Somfai (BB.)

 and Denijs Dille (DD.)

 are provided, as well as Bartók's own opus numbers.

 Note that Bartók started three times anew with opus numbers, here indicated with "(list 1)", "(list 2)" and "(list 3)" respectively.

 The pieces from the third listing are by far best known; opus lists 1 and 2 are early works.

 The year of composition and instrumentation (including voice) are included.

 See the main article on Béla Bartók for more details.
</item 10>
=== END ITEM id=79b6a08b4fc97ca0 ===

=== ITEM id=6510bdb3d917f249 ===
Gold sentences and facts:
Sentence 0: The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
  Fact 0.0: The Silent Historian has the original title 'Het zwijgen van Loe de Jong [The Silence of Loe de Jong]'.
  Fact 0.1: The Silent Historian is a 2011 Dutch documentary film.
  Fact 0.2: The Silent Historian is a documentary film by Simonka de Jong.
  Fact 0.3: The Silent Historian was shown as part of the January 2012 New York Jewish Film Festival.
  Fact 0.4: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Jewish Museum.
  Fact 0.5: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Film Society of Lincoln Center.
Sentence 1: The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
  Fact 1.0: The Betrayal — Nerakhoon is a 2008 documentary film.
  Fact 1.1: The Betrayal — Nerakhoon was directed by Ellen Kuras.
  Fact 1.2: The Betrayal — Nerakhoon was directed by Thavisouk Phrasavath.

Retrieved items:
<item 1>
Betrayal (1929 film)
Betrayal was a 1929 drama film produced for Famous Players-Lasky and released by Paramount Pictures.

 The film is the last silent film directed by Lewis Milestone, the last silent performance by Gary Cooper, the last silent performance by Germany's Emil
Jannings, and the only onscreen pairing of Cooper and Jannings.

 It is considered a lost film.
</item 1>

<item 2>
Graph query row:
film: The Silent Historian
film_type: documentary film
genre: None
linked_genre: None
</item 2>

<item 3>
The Decembrists (film)
The Decembrists (Russian: Декабристы , "Dekabristy " ) is a 1927 Soviet silent historical drama film directed by Aleksandr Ivanovsky.
</item 3>

<item 4>
The Price of Betrayal
The Price of Betrayal (Swedish: "Judaspengar" ) is a 1915 Swedish silent drama film directed by Victor Sjöström.
</item 4>

<item 5>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 5>

<item 6>
The Counterfeiters (2007 film)
The Counterfeiters (German: "Die Fälscher" ) is a 2007 Austrian-German drama film written and directed by Stefan Ruzowitzky.

 It fictionalizes Operation Bernhard, a secret plan by Nazi Germany during World War II to destabilize the United Kingdom by flooding its economy with forged
Bank of England pound notes.

 The film centres on a Jewish counterfeiter, Salomon 'Sally' Sorowitsch, who is coerced into assisting the operation at the Sachsenhausen concentration camp.
</item 6>

<item 7>
Graph query row:
film: The Betrayal – Nerakhoon
film_type: documentary film
genre: None
linked_genre: None
</item 7>

<item 8>
Thérèse Raquin (1928 film)
Thérèse Raquin is a 1928 drama film directed by Jacques Feyder.

 It is the third silent film adaptation of the novel of the same name by Émile Zola.

 The film stars Gina Manès as Thérèse Raquin, Wolfgang Zilzer as Monsieur Raquin, and Jeanne Marie Laurent as Madame Raquin.

 The décors of the Paris suburbs for the film were built by André Andrejew.

 The film was produced by Deutsche Film Union in Germany, with German and French actors, in a French-German co-production, to be later released at the same time
in France as "Thérèse Raquin" and Germany as Du sollst nicht ehebrechen!

.

 As no words were spoken, both versions differed only with the language of intertitles.

 The British title at the time of the film's original release was Thou Shalt Not.

 This is last silent film imports distributed by Warner Bros.' newly acquired First National subsidiary.

 No dialong with music score and sound effects.
</item 8>

<item 9>
The Silent Historian
The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as
part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
</item 9>

<item 10>
The Betrayal – Nerakhoon
The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
</item 10>
=== END ITEM id=6510bdb3d917f249 ===

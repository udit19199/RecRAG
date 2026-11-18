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

=== ITEM id=c35701ea349c6032 ===
Gold sentences and facts:
Sentence 0: The Pan-American Highway is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) , called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World Records , the Pan-American Highway is the world 's longest `` motorable road '' . However , because of the Darién Gap , it is not possible to cross between South America and Central America with conventional highway vehicles . Without an all - terrain vehicle , it is necessary to circumnavigate this terrestrial stretch by sea .
  Fact 0.0: The Pan-American Highway is a network of roads.
  Fact 0.1: The Pan-American Highway measures about 30,000 kilometres (19,000 mi) in total length.
  Fact 0.2: The Pan-American Highway has a rainforest break of approximately 160 km (100 mi).
  Fact 0.3: The rainforest break of the Pan-American Highway is called the Darién Gap.
  Fact 0.4: Apart from the Darién Gap, the Pan-American Highway links almost all of the Pacific coastal countries of the Americas.
  Fact 0.5: The Pan-American Highway links those countries in a connected highway system.
  Fact 0.6: According to Guinness World Records, the Pan-American Highway is the world's longest 'motorable road'.
  Fact 0.7: Because of the Darién Gap, it is not possible to cross between South America and Central America with conventional highway vehicles.
  Fact 0.8: Without an all-terrain vehicle, it is necessary to go around the terrestrial stretch of the Darién Gap by sea.

Retrieved items:
<item 1>
Pan-American Highway
They barely succeeded in thrashing a passage through the extreme terrain . In 1979 a team led by Mark Smith drove standard production CJ7 - model Jeeps from
South to North , traversing the Gap - with difficulty . In June 1984 , Loren and Patty Upton took 741 days slogging , winching , chopping and digging their way
through the inhospitable jungles of the Darién Gap . One proposed option to bridge the gap is a short ferry link from Colombia to a new ferry port in Panama ,
with an extension of the existing Panama highway that would complete the highway without violating these environmental concerns . Southern section ( edit ) Main
article : Pan-American Highway ( South America ) A Vía PanAm shield sign is sometimes found on routes in South American countries ( such as Chile ) associated
with the Pan-American Highway . Sculpture of a native man standing at the entrance of Fusagasugá , Colombia , over the Highway 40 . Colombia and Venezuela (
edit ) The southern part of the highway begins in northwestern Colombia , from where it follows Colombia Highway 62 to Medellín .
</item 1>

<item 2>
Pan-American Highway
Panamericana -- Pan American Highway -- in Pichincha , Ecuador , near Cashapamba Panamericana -- northern Peru near Pacasmayo Panamericana near Puerto De Lomas
, Peru Panamericana in the Atacama Desert northern Chile GMC Suburban traveled all of the Pan-American Highway . Patagonia , Chile . Panamericana - Pan American
Highway - in the Cordillera de los Andes , southern Ecuador , near to Catacocha , 2500m / 8000 ft Elev . National Route 7 -- Pan-American Highway in Junín ,
Buenos Aires , Argentina See also ( edit ) CANAMEX Corridor Continental 1 Trans - Siberian Highway Buenos Aires References ( edit ) Jump up ^ French : Route /
Autoroute Panaméricaine / Transaméricaine , Portuguese : Rodovia / Auto - estrada Pan-americana , Spanish : Autopista / Carretera / Ruta Panamericana Jump up ^
`` A Gap in the Andes : Image of the Day '' . earthobservatory.nasa.gov. 2015 - 04 - 02 .
</item 2>

<item 3>
Pan-American Highway
Pan-American Highway at David , Chiriquí The Pan-American ( or Inter-American ) highway passes through the Central American countries with the highway
designation of CA - 1 ( Central American Highway 1 ) . In Guatemala , it passes through 10 departments , including The Department Of Guatemala , where it passes
through Guatemala City . In El Salvador , it passes through the cities of Santa Ana , Santa Tecla , Antiguo Cuscatlán , San Salvador , San Martín , San Miguel ,
and crosses the border into Honduras at Amatillo . From Honduras , it passes into Nicaragua , passing though the Nicaraguan cities of Chinandega , Estelí ,
Sebaco Matagalpa , León , and Managua , before entering Costa Rica at Peñas Blancas . In Costa Rica , it passes through Liberia , San José , Cartago , Pérez
Zeledón , Palmares , Neily , before crossing into Panama at Paso Canoas . In Panama , it crosses the Panama Canal via the Centennial Bridge , and ends at Yaviza
, at the edge of the Darién Gap .
</item 3>

<item 4>
Pan-American Highway
Several southern highway termini are claimed to exist , including the cities of Puerto Montt and Quellón in Chile and Ushuaia in Argentina . West and north of
the Darien Gap , it is also known as the Inter-American Highway through Central America and Mexico where it splits into several spurs leading to the Mexico --
US border . The entire Interstate Highway System in the United States has been officially designated part of the Pan American system by the Federal Highway
Administration , although this has never been reflected in any of the official interstate signage . Of the many freeways that make up this very comprehensive
system , there are several that stand out because of their mainly north - south orientation and their links to the main Mexican route and its spurs . These
include the following : Interstate 8 indirectly connects Interstate 19 and its Mexican spur at Nogales with I - 5 and Interstate 15 at San Diego , each of which
leads on separate routes from the latter city towards the Alaska Highway . The north end of I - 5 links indirectly with British Columbia Highway 97 north of the
Canada -- US border . BC 97 becomes the Alaska Highway at Dawson Creek .
</item 4>

<item 5>
Pan-American Highway
Contents ( hide ) 1 Development and construction 2 Countries served 3 Hemispheric overview 4 Northern section 4.1 Alaska , Canada and the contiguous 48 states
of the United States 4.2 Mexico 4.3 Central America 5 Darién Gap 6 Southern section 6.1 Colombia and Venezuela 6.2 Ecuador , Peru , Chile , Argentina and
Paraguay 6.3 Brazil and Uruguay 6.4 Guyana , Suriname and French Guyana 7 West Indies section 8 Art and culture 9 Photo gallery 10 See also 11 References 12
Sources 13 External links Development and construction ( edit ) The concept of an overland route from one tip of the Americas to the other was originally
proposed at the First Pan-American Conference in 1889 as a railroad ; however , this proposal was never realized . The idea of building a highway emerged at the
Fifth International Conference of American States in 1923 . The first conference regarding construction of the highway occurred on October 5 , 1925 .
</item 5>

<item 6>
Pan-American Highway
In 2003 , Kevin Sanders , a long - distance rider , broke the Guinness World Record for the fastest traversal of the highway by motorcycle in 34 days . Photo
gallery ( edit ) The northern end of the Pan-American Highway at Deadhorse , Alaska Interstate 25 ( Pan-American Freeway ) approaching the Big I interchange in
Albuquerque , New Mexico , USA Pan-American Highway through San Martin , El Salvador . Another view of the Pan-American Highway in El Salvador . Pan-American
Highway in El Salvador between Lourdes and Santa Ana ; this flat 1.5 km long straight section can be used as an airstrip and it was used during El Salvador
Civil War . Highway in Guanacaste , Costa Rica ( going towards the Nicaraguan border , still many kilometres away . ) Pan-American Highway in Tres Rios , Costa
Rica , right before the toll plaza ( about 337 more km to go until the Panamanian border ) .
</item 6>

<item 7>
Pan-American Highway
Pan-American Highway - wikipedia Pan-American Highway Jump to : navigation , search `` Panamericana '' redirects here . For the film , see Panamericana ( film )
. Not to be confused with CANAMEX Corridor . The Pan-American Highway from Prudhoe Bay , U.S.A. to Quellón , Chile and Ushuaia , Argentina , with official and
unofficial routes shown in South America , and a few selected unofficial routes shown through the United States and Canada as they existed in the early 1960 's
. In 1966 the new U.S. Interstate highway system brought official status to most previously unofficial routes in the lower 48 states . The Pan-American Highway
is a network of roads measuring about 30,000 kilometres ( 19,000 mi ) in total length . Except for a rainforest break of approximately 160 km ( 100 mi ) ,
called the Darién Gap , the road links almost all of the Pacific coastal countries of the Americas in a connected highway system . According to Guinness World
Records , the Pan-American Highway is the world 's longest `` motorable road '' .
</item 7>

<item 8>
Pan-American Highway
In Canada , Ontario Highway 402 and other freeways in the Windsor - Quebec City Corridor can be considered a northeastward extension of this version the NAFTA
superhighway . To the southwest , from western Kentucky to the Mexican border , there is currently no single superhighway yet completed . Pending completion of
I - 69 , the main highway links to Mexico follow parts of US routes 45 and 51 from Kentucky to western Tennessee , I - 155 into Missouri , parts of Interstates
55 and 40 from Missouri to Arkansas , and I - 30 to the Texas stretch of I - 35 that continues to the Mexican border at Laredo , Texas . The incompleted section
of I - 69 south of Kentucky is expected to eventually continue southwestward to the Texas Gulf Coast . It will have a spur linking to the original Pan-American
route through Mexico to Laredo , and additional branches extending to the Mexican spurs that cross the border at Pharr , Texas , and Brownsville , Texas .
Mexico ( edit ) 1933 map of the Inter-American Highway portion of the Pan-American Highway .
</item 8>

<item 9>
Pan-American Highway
The official route of the Pan-American Highway through Mexico ( where it is known as the Inter-American Highway ) starts at Nuevo Laredo , Tamaulipas ( opposite
Laredo , Texas ) and goes south to Mexico City along Mexican Federal Highway 85 . Later branches were built to the border as follows : Nogales spur -- Mexican
Federal Highway 15 from Mexico City El Paso spur -- Mexican Federal Highway 45 from Highway 85 north of Mexico City to Ciudad Juárez , Chihuahua ) Eagle Pass
spur -- unknown , possibly Mexican Federal Highway 57 from Mexico City to Piedras Negras , Coahuila Pharr spur -- Mexican Federal Highway 40 from Monterrey to
Reynosa , Tamaulipas Brownsville spur -- Mexican Federal Highway 101 from Ciudad Victoria to Matamoros , Tamaulipas From Mexico City to the border with
Guatemala , the highway follows Mexican Federal Highway 190 . Central America ( edit ) Pan-American Highway in Chimaltenango ( Guatemala ) , 2001 .
</item 9>

<item 10>
Pan-American Highway
Travelers would then be able to ferry from Key West to Havana , drive to the eastern tip of Cuba , ferry to Haiti , drive through Haiti and the Dominican
Republic , and ferry again to Puerto Rico . Included in this system would also be a ferry from the western tip of Cuba to the Yucatán Peninsula . Mexico has
already surveyed a route which will run across the Yucatán , Campeche , and Chiapas to San Cristobal de Las Casas , on the Pan American Highway . ( `` The Pan
American Highway System '' by Travel Division Pan American Union , Washington D.C. October 1947 ) Art and culture ( edit ) Travel writer Tim Cahill wrote a book
, Road Fever , about his record - setting 24 - day drive from Ushuaia in the Argentine province of Tierra del Fuego to Prudhoe Bay in the U.S. state of Alaska
with professional long - distance driver Garry Sowerby , much of their route following the Pan-American Highway . In the British motoring show Top Gear , the
presenters drove on a section of the road in their off - road vehicles in the Bolivian Special .
</item 10>
=== END ITEM id=c35701ea349c6032 ===

=== ITEM id=64661f6c21d83366 ===
Gold sentences and facts:
Sentence 0: Your Lie in April Cover of the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono 四 月 は 君 の 嘘 ( Shigatsu wa Kimi no Uso ) Manga Written by Naoshi Arakawa Published by Kodansha English publisher Kodansha Comics Demographic Shōnen Magazine Monthly Shōnen Magazine Original run April 6 , 2011 -- February 6 , 2015 Volumes 11 ( List of volumes ) Anime television series Directed by Kyōhei Ishiguro Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1 Pictures Licensed by Madman Entertainment Aniplex of America Anime Limited Original network Fuji TV ( Noitamina ) Original run October 9 , 2014 -- March 19 , 2015 Episodes 22 ( List of episodes ) Light novel Your Lie in April - A Six Person Etude Written by Yui Tokiumi Published by Kodansha English publisher Vertical Published November 17 , 2014 Volumes Manga Shigatsu wa Kimi no Uso : Coda Written by Naoshi Arakawa Demographic Shōnen Original run February 25 , 2015 -- October 28 , 2015 Volumes Original video animation Moments Directed by Kazuya Iwata Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1 Pictures Released May 15 , 2015 Runtime 23 minutes Live - action film Your Lie in April ( film ) Anime and Manga portal
  Fact 0.0: The cover shows the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono.
  Fact 0.1: The title is 四 月 は 君 の 嘘 (Shigatsu wa Kimi no Uso).
  Fact 0.2: The manga Shigatsu wa Kimi no Uso is written by Naoshi Arakawa.
  Fact 0.3: The manga is published by Kodansha, with English publisher Kodansha Comics.
  Fact 0.4: The manga has the Shōnen demographic and was serialized in Monthly Shōnen Magazine.
  Fact 0.5: The manga's original run was April 6, 2011 to February 6, 2015.
  Fact 0.6: The manga has 11 volumes.
  Fact 0.7: The anime television series is directed by Kyōhei Ishiguro.
  Fact 0.8: The anime television series is written by Takao Yoshioka.
  Fact 0.9: The anime television series has music by Masaru Yokoyama.
  Fact 0.10: The anime television series is by studio A-1 Pictures.
  Fact 0.11: The anime is licensed by Madman Entertainment and Aniplex of America Anime Limited.
  Fact 0.12: The anime's original network is Fuji TV (Noitamina).
  Fact 0.13: The anime's original run was October 9, 2014 to March 19, 2015.
  Fact 0.14: The anime has 22 episodes.
  Fact 0.15: The light novel Your Lie in April - A Six Person Etude is written by Yui Tokiumi.
  Fact 0.16: The light novel is published by Kodansha, with English publisher Vertical.
  Fact 0.17: The light novel was published November 17, 2014.
  Fact 0.18: The manga Shigatsu wa Kimi no Uso: Coda is written by Naoshi Arakawa.
  Fact 0.19: The manga Shigatsu wa Kimi no Uso: Coda has the Shōnen demographic.
  Fact 0.20: The manga Shigatsu wa Kimi no Uso: Coda's original run was February 25, 2015 to October 28, 2015.
  Fact 0.21: The original video animation Moments is directed by Kazuya Iwata.
  Fact 0.22: The original video animation Moments is written by Takao Yoshioka.
  Fact 0.23: The original video animation Moments has music by Masaru Yokoyama.
  Fact 0.24: The original video animation Moments is by studio A-1 Pictures.
  Fact 0.25: The original video animation Moments was released May 15, 2015.
  Fact 0.26: The original video animation Moments has a runtime of 23 minutes.
  Fact 0.27: There is a live-action film titled Your Lie in April.

Retrieved items:
<item 1>
Your Lie in April
The letter states that she was in love with Kōsei the whole time , and that her supposed feelings towards Watari were fabricated in order to get closer to Kōsei
without emotionally injuring Tsubaki , who also has limerent feelings towards Kōsei . Although at first consumed with grief , Kōsei finds comfort in Tsubaki ,
who promises not to abandon him . Characters ( edit ) Kōsei Arima ( 有馬 公 生 , Arima Kōsei ) Voiced by : Natsuki Hanae ( Japanese ) ; Max Mittelman ( English ) &
Piano performance by : Tomoki Sakata Played by : Kento Yamazaki ( live - action film ) ; Shintarō Anzai ( stage play ) Kōsei is a former child prodigy in
playing piano , dubbed the `` Human Metronome '' for his near - inhuman mechanical accuracy , a product of his mother 's overly strict methods of teaching . His
ability to play the piano with unparalleled precision led him to win many competitions across Japan and even be invited to play abroad .
</item 1>

<item 2>
Your Lie in April
. KODANSHA COMICS . Retrieved February 23 , 2015 . Jump up ^ `` Vertical to Release Your Lie in April Light Novel in July '' . Anime News Network . May 22 ,
2017 . Retrieved May 23 , 2017 . Jump up ^ `` Shōsetsu Shigatsu wa Kimi no Uso - 6 - nin no Etude '' 小説 四 月 は 君 の 嘘 6 人 の エチュード . Kodansha . Retrieved May 23 ,
2017 . Jump up ^ `` Shigatsu wa Kimi no Uso Manga Gets Noitamina Anime '' . Anime News Network . March 21 , 2014 . Retrieved March 21 , 2014 . ^ Jump up to : ``
Live - Action Your Lie in April Film Opens in September '' . Anime News Network . Retrieved 29 January 2016 .
</item 2>

<item 3>
Your Lie in April
It premiered on October 10 , 2014 in Japan and ended on March 20 , 2015 . The first opening theme song is `` Hikaru Nara '' ( 光る なら , lit . If You Will Shine )
by Goose house and the first ending theme song is `` Kirameki '' ( キラメキ , lit . Sparkle ) by wacci . The second opening song is `` Nanairo Symphony '' ( 七色
シンフォニー , Nanairo Shinfonī , lit . `` Seven Colored Symphony '' ) by Coalamode and the second ending theme is `` Orange '' ( オレンジ , Orenji ) by 7 ! ! ( Seven
Oops ) . The original soundtracks are composed by Masaru Yokoyama . In North America , Aniplex of America licensed the series within the region , who
simulcasted the series on Crunchyroll , Aniplex Channel , Hulu and Viewster . In Australia and New Zealand , the series is licensed by Madman Entertainment ,
who streamed the series on AnimeLab .
</item 3>

<item 4>
Your Lie in April
Your Lie in April - Wikipedia Your Lie in April Jump to : navigation , search This article is about the manga and anime . For the live - action film adaptation
, see Your Lie in April ( film ) . Your Lie in April Cover of the first tankōbon volume featuring Kōsei Arima and Kaori Miyazono 四 月 は 君 の 嘘 ( Shigatsu wa Kimi
no Uso ) Manga Written by Naoshi Arakawa Published by Kodansha English publisher Kodansha Comics Demographic Shōnen Magazine Monthly Shōnen Magazine Original
run April 6 , 2011 -- February 6 , 2015 Volumes 11 ( List of volumes ) Anime television series Directed by Kyōhei Ishiguro Written by Takao Yoshioka Music by
Masaru Yokoyama Studio A-1 Pictures Licensed by Madman Entertainment Aniplex of America Anime Limited Original network Fuji TV ( Noitamina ) Original run
October 9 , 2014 -- March 19 ,
</item 4>

<item 5>
Your Lie in April
2014 -- March 19 , 2015 Episodes 22 ( List of episodes ) Light novel Your Lie in April - A Six Person Etude Written by Yui Tokiumi Published by Kodansha English
publisher Vertical Published November 17 , 2014 Volumes Manga Shigatsu wa Kimi no Uso : Coda Written by Naoshi Arakawa Demographic Shōnen Original run February
25 , 2015 -- October 28 , 2015 Volumes Original video animation Moments Directed by Kazuya Iwata Written by Takao Yoshioka Music by Masaru Yokoyama Studio A-1
Pictures Released May 15 , 2015 Runtime 23 minutes Live - action film Your Lie in April ( film ) Anime and Manga portal Your Lie in April , known in Japan as
Shigatsu wa Kimi no Uso ( 四 月 は 君 の 嘘 ) or just simply Kimiuso , is a Japanese manga series written and illustrated by Naoshi Arakawa . The series was
serialized in Kodansha 's Monthly Shōnen Magazine from April 2011 to May 2015 .
</item 5>

<item 6>
Your Lie in April
Kaori Miyazono ( 宮園 かをり , Miyazono Kawori ) Voiced by : Risa Taneda ( Japanese ) ; Erica Lindbeck ( English ) & Violin performance by : Yūna Shinohara Played by
: Suzu Hirose ( live - action film ) ; Arisa Matsunaga ( stage play ) Kaori is Tsubaki 's classmate ; she is a free - spirited violinist who has drawn much
criticism from judging panels due to her unwillingness to adhere strictly to the score , but is highly favored by audiences that hear her playing . Kaori first
met Kōsei when she asked Tsubaki to set her up with Watari . As their friendship grew , she eventually convinced Kōsei to play the piano again , first as her
accompanist and later in a piano competition . It is revealed at the end of the anime , in a letter addressed to Kōsei , that she had asked to be set up with
Watari in order to meet Kōsei , knowing that Watari and Tsubaki were good friends of his .
</item 6>

<item 7>
Your Lie in April
Media ( edit ) Manga ( edit ) The Your Lie in April manga series is written and illustrated by Naoshi Arakawa , and began serialization on April 6 , 2011 in
Kodansha 's Monthly Shōnen Magazine . The first tankōbon was published by Kodansha on September 16 , 2011 , and the final volume was released in two editions ,
a regular edition and a limited edition , with the limited edition bundled with an OVA , and was released on May 15 , 2015 . The series is licensed in English
in North America by Kodansha Comics , and the first volume was published on April 21 , 2015 . A spin - off manga titled Shigatsu wa Kimi no Uso : Coda was
bundled with the release of the Blu - ray release of the anime series . It was later published in tankōbon format on August 17 , 2016 . Volume list ( edit ) No
. Japanese release date Japanese ISBN English release date English ISBN September 16 , 2011 ISBN 978 - 4 - 06 - 371301 - 5 April 21 ,
</item 7>

<item 8>
Your Lie in April
An anime television series adaptation by A-1 Pictures aired from October 2014 to March 2015 on Fuji TV 's Noitamina block . A live - action film adaptation of
the same name was released in September 2016 . Contents ( hide ) 1 Plot 2 Characters 3 Media 3.1 Manga 3.1. 1 Volume list 3.2 Anime 3.3 Live - action film 3.4
Stage play 4 Reception 5 References 6 External links Plot ( edit ) Piano prodigy Kōsei Arima dominated various music competitions and has become famous among
child musicians , but also controversial . After the death of his abusive mother , who was also his coldhearted instructor who forced him to play the piano
emotionlessly , he had a mental breakdown at the age of eleven while performing at a piano recital which resulted in him no longer being able to hear the sound
of his piano even though his hearing was perfectly fine .
</item 8>

<item 9>
Your Lie in April
When his mother died , the resulting psychological trauma caused him to be unable to hear the sound of his piano playing , and he gave up on it . Two years
later , he takes up the piano again after being convinced by Kaori Miyazono to become her accompanist . Influenced by her emotional and unrestrained playing
style , Kōsei eventually finds himself falling in love with her . He , however , does not confess his feelings due to her liking his best friend , Ryota Watari
. It is later revealed in Shigatsu wa Kimi no Uso : Coda , a side story from Kōsei and Tsubaki 's childhood , that his inspiration for playing so beautifully in
his first competition was to cheer up Tsubaki after the passing of her grandmother .
</item 9>

<item 10>
Your Lie in April
Kodansha . Retrieved January 27 , 2016 . Jump up ^ `` Your Lie in April 9 '' . Penguin Random House . Retrieved April 11 , 2016 . Jump up ^ 四 月 は 君 の 嘘 ( 10 ) (
Your Lie in April ( 10 ) ) ( in Japanese ) . Kodansha . Retrieved January 27 , 2016 . Jump up ^ `` Your Lie in April 10 '' . Penguin Random House . Retrieved
November 6 , 2016 . Jump up ^ `` Your Lie in April 11 '' . Penguin Random House . Retrieved April 11 , 2016 . Jump up ^ `` Goose House to Perform Opening ,
Cover Songs for Shigatsu wa Kimi no Uso Anime '' . Anime News Network . August 15 , 2014 . Retrieved August 3 , 2017 . Jump up ^ `` Coala Mode , 7 ! ! Perform
Theme Songs For Your Lie in April Anime 's 2nd Half '' . Anime News Network .
</item 10>
=== END ITEM id=64661f6c21d83366 ===

=== ITEM id=492993c0be56a063 ===
Gold sentences and facts:
Sentence 0: Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning , genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through artificial insemination .
  Fact 0.0: Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse.
  Fact 0.1: Superfecundation can lead to twin babies from two separate biological fathers.
  Fact 0.2: The term superfecundation is derived from fecund.
  Fact 0.3: Fecund means the ability to produce offspring.
  Fact 0.4: Heteropaternal superfecundation refers to the fertilization of two separate ova by two different fathers.
  Fact 0.5: Homopaternal superfecundation refers to the fertilization of two separate ova from the same father.
  Fact 0.6: Homopaternal superfecundation leads to fraternal twins.
  Fact 0.7: Heteropaternal superfecundation is referred to as a form of atypical twinning.
  Fact 0.8: Genetically, the twins from heteropaternal superfecundation are half siblings.
  Fact 0.9: Superfecundation is rare.
  Fact 0.10: Superfecundation can occur through separate occurrences of sexual intercourse.
  Fact 0.11: Superfecundation can occur through artificial insemination.

Retrieved items:
<item 1>
Superfecundation
This led to the discovery that the twins had different fathers . In 1982 , twins who were born with two different skin colors were discovered to be conceived as
a result of heteropaternal superfecundation . In 2001 , a case of spontaneous monopaternal superfecundation was reported after a woman undergoing IVF treatments
gave birth to quintuplets after only two embryos were implanted . Genetic testing supported that the twinning was not a result of the embryos splitting , and
that all five boys shared the same father . In 2016 , an IVF - implanted surrogate mother gave birth to two children : one genetically unrelated child from an
implanted embryo , and a biological child from her own egg and her husband 's sperm . See also ( edit ) Castor and Pollux , said to be twins having different
fathers Chimera ( genetics ) Mixed twins Polyspermy References ( edit ) Jump up ^ Worland , Justin ( May 8 , 2015 ) . `` The Science of How Women Can Have Twins
With 2 Different Fathers '' . Time . Retrieved March 23 , 2017 .
</item 1>

<item 2>
Superfecundation
This is known as superfetation . Heteropaternal superfecundation in mammals ( edit ) Heteropaternal superfecundation is common in animals such as cats and dogs
. Stray dogs can produce litters in which every puppy has a different sire . Though rare in humans , cases have been documented . In one study on humans , the
frequency was 2.4 % among dizygotic twins whose parents had been involved in paternity suits . Cases involving superfecundation ( edit ) Koen and Teun Stuart ,
Dutch boys who were the result of in vitro fertilization ( IVF ) . In a mixup at the laboratory , equipment had been used twice , causing another man 's sperm
to be mixed with the intended father 's . In 1995 , a young woman gave birth to diamniotic monochorionic twins , who were originally assumed to be monozygotic
twins until a paternity suit led to a DNA test . This led to the discovery that the twins had different fathers .
</item 2>

<item 3>
Superfecundation
Fertility and Sterility. 76 ( 3 ) : 621 -- 3 . doi : 10.1016 / S0015 - 0282 ( 01 ) 01976 - 8 . PMID 11532493 . Jump up ^ Susan Scutti ( November 3 , 2017 ) . ``
Surrogate mom delivers two babies , one her own '' . CNN . Further reading ( edit ) `` Doctors Discover Dallas Twins Have Two Different Dads '' , Fox News , Sky
News , 2009 Lu , H.L. ; Wang , C.X. ; Wu , F.Q. ; Li , J.J. ( 1994 ) . `` Paternity identification in twins with different fathers '' . Journal of forensic
sciences . 39 ( 4 ) : 1100 -- 2 . PMID 8064269 .
</item 3>

<item 4>
Superfecundation occurs when separate eggs are fertilized in the same cycle. Heteropaternal superfecundation can produce twins with different fathers; a 1982
case involved twins with different skin colors. Monopaternal superfecundation—multiple fertilizations by the same father—was reported in 2001 when IVF resulted
in quintuplets after two embryos were transferred. In 2016, a surrogate gave birth to an implanted embryo’s child and a second child
</item 4>

<item 5>
Superfecundation
PMID 26548710 . Jump up ^ Wenk , R.E. ; Houtz , T ; Brooks , M ; Chiafari , F.A. ( 1992 ) . `` How frequent is heteropaternal superfecundation ? '' . Acta
geneticae medicae et gemellologiae. 41 ( 1 ) : 43 -- 7 . PMID 1488855 .
</item 5>

<item 6>
Superfecundation
Time . Retrieved March 23 , 2017 . Jump up ^ `` Public Education Pamphlets '' . sogc.org . ^ Jump up to : Ambach , E ; Parson , W ; Brezinka , C ( 2000 ) . ``
Superfecundation and dual paternity in a twin pregnancy ending with placental abruption '' . Journal of forensic sciences . 45 ( 1 ) : 181 -- 3 . PMID 10641935
. ^ Jump up to : McNamara , Helen C. ; Kane , Stefan C. ; Craig , Jeffrey M. ; Short , Roger V. ; Umstad , Mark P. ( 2016 ) . `` A review of the mechanisms and
evidence for typical and atypical twinning '' . American Journal of Obstetrics and Gynecology. 214 ( 2 ) : 172 -- 91 . doi : 10.1016 / j. ajog. 2015.10. 930 .
PMID 26548710 . Jump up ^ Wenk , R.E.
</item 6>

<item 7>
Superfecundation
Superfecundation - wikipedia Superfecundation Jump to : navigation , search This article needs additional citations for verification . Please help improve this
article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2009 ) ( Learn how and when to remove this template
message ) Superfecundation is the fertilization of two or more ova from the same cycle by sperm from separate acts of sexual intercourse , which can lead to
twin babies from two separate biological fathers . The term superfecundation is derived from fecund , meaning the ability to produce offspring . Heteropaternal
superfecundation refers to the fertilization of two separate ova by two different fathers . Homopaternal superfecundation refers to the fertilization of two
separate ova from the same father , leading to fraternal twins . While heteropaternal superfecundation is referred to as a form of atypical twinning ,
genetically , the twins are half siblings . Superfecundation , while rare , can occur through either separate occurrences of sexual intercourse or through
artificial insemination .
</item 7>

<item 8>
Superfecundation occurs when two or more eggs released in one menstrual cycle are fertilized by sperm from separate acts of intercourse or artificial
insemination. If the sperm come from different fathers, the fraternal twins are genetic half-siblings; if from the same father, they are ordinary fraternal
twins. Eggs remain fertilizable for 12–48 hours, while sperm can survive up to five days. Heteropaternal superfecundation is common in some animals but rare in
humans;
</item 8>

<item 9>
Superfecundation
PMID 8064269 . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Superfecundation&oldid=818981440 '' Categories : Reproduction Twin Triplets Multiple
births Free sex Sibling Hidden categories : Articles needing more detailed references All articles with dead external links Articles with dead external links
from June 2017 Articles needing additional references from May 2009 All articles needing additional references Talk Contents About Wikipedia Deutsch فارسی
Français Nederlands Polski Português Русский Slovenčina Edit links This page was last edited on 6 January 2018 , at 18 : 28 . About Wikipedia
</item 9>

<item 10>
Superfecundation
PMID 1488855 . Jump up ^ Fierro , Pamela Prindle ( January 5 , 2016 ) , `` Heteropaternal Superfecundation - Twins with Different Fathers '' , About.com ,
archived from the original on April 4 , 2016 Jump up ^
https://www.scopus.com/record/display.uri?eid=2-s2.0-0020008620&origin=inward&txGid=F6227767DB3CDB724EA92EFC60A5A985.wsnAw8kcdt7IPYLO0V48gA%3a2 Jump up ^
Amsalem , Hagai ; Tsvieli , Rimona ; Zentner , Bat Sheva ; Yagel , Simcha ; Mitrani - Rosenbaum , Stella ; Hurwitz , Arye ( 2001 ) . `` Monopaternal
superfecundation of quintuplets after transfer of two embryos in an in vitro fertilization cycle '' . Fertility and Sterility. 76 ( 3 ) : 621 -- 3 .
</item 10>
=== END ITEM id=492993c0be56a063 ===

=== ITEM id=def2e96b1e7b767b ===
Gold sentences and facts:
Sentence 0: Teri Garr as Mrs. Ramsay , Alec 's mother
  Fact 0.0: Teri Garr played the role of Mrs. Ramsay
  Fact 0.1: Mrs. Ramsay is Alec's mother

Retrieved items:
<item 1>
The Black Stallion (film)
1 John Buchanan as Jockey No. 2 Kristen Vigard as Becky Fausto Tozzi as Rescue Captain John Karlsen as Archeologist Leopoldo Trieste as Priest Frank Cousins as
African Chieftain Don Hudson as Zaurog Marne Maitland as Drake Captain Tom Dahlgren as Veterinarian Horses ( edit ) Cass Ole , a champion Arabian stallion , was
featured in most of the movie 's scenes , with Fae Jur , another black Arabian stallion , being his main double . Fae Jur 's main scene is the one where Alec is
trying to gain the trust of the Black on the beach . Two other stunt doubles were used for running , fighting and swimming scenes . El Mokhtar , an Egyptian
Arabian racehorse , was the producers ' first choice to portray the Black , but they were unable to secure his services for the film from his owners , who
declined any offers . He does appear in The Black Stallion Returns , alongside Cass Ole , by which time the studio bought out the syndicate of owners in order
to secure El Mokhtar 's services .
</item 1>

<item 2>
The Black Stallion (film)
Alec 's father rescues Alec from his attacker but the life jacket is lost and father & son are separated . The black stallion breaks free from his compartment
and then jumps into the sea . Alec himself is swept overboard by a gigantic wave . Once in the water , he swims toward the stallion and managed to grab hold of
the ropes that were previously restraining the stallion . Alec wakes on the shore of a deserted island and starts to explore . He finds the stallion caught in
his restraints with the ropes stuck between the rocks . With his knife , Alec manages to free the stallion once again and the stallion runs away . For a time ,
the two keep their distance . Alec discovers means to survive by catching fish and seaweed . As Alec suddenly faces a cobra eye to eye , the Black comes to the
rescue and kills the snake , only to run off again . By now , Alec decides to try to get closer to the horse and offer him some seaweed . The hungry stallion
finds himself unable to resist , but visibly struggles with his distrust for humans .
</item 2>

<item 3>
The Black Stallion (film)
The hungry stallion finds himself unable to resist , but visibly struggles with his distrust for humans . Eventually , the hunger wins and he takes Alec 's
offer ; their bond has been sealed and the two are now inseparable . Alec even manages to ride the unbroken horse , after many times falling off the horse . One
day , a fishing ship arrives , rescuing both Alec and the stallion . Back home in Flushing , NY , Alec is given a hero 's welcome . The Black has a temporary
home in Alec 's backyard , but a garbage man not knowing that there is a wild horse in the backyard is chased by the Black , who races off down the street after
being spooked by a passing car . Alec chases after him through every part of town , but loses track of him . The next day , Alec meets Snoe ( and Napoleon ) who
tell him where the Black is . Alec finds the stallion in the barn of Henry Dailey , a retired jockey and racehorse trainer , who apparently spent all night
catching the Black . Henry allows Alec to keep the Black in his barn .
</item 3>

<item 4>
The Black Stallion (film)
He falls behind , but the Black wo n't stop . When Alec regains his balance , the Black is well on his way to catch up with his opponents . Alec now encourages
the Black to run as fast as he can , remembering the wild rides on the island , as they catch up . The Black eventually wins by two lengths . Cast ( edit )
Kelly Reno as Alec Ramsay Mickey Rooney as Henry Dailey Teri Garr as Mrs. Ramsay , Alec 's mother Hoyt Axton as Mr. Ramsay , Alec 's father Clarence Muse as
Snoe Michael Higgins as Jim Neville Ed McNamara as Jake Doghmi Larbi as Arab John Burton as Jockey No. 1 John Buchanan as Jockey No.
</item 4>

<item 5>
The Black Stallion (film)
American Film Institute . Retrieved 2016 - 08 - 20 . External links ( edit ) Official website The Black Stallion on IMDb The Black Stallion at AllMovie The
Black Stallion at Rotten Tomatoes The Black Stallion at Box Office Mojo The Black Stallion at the TCM Movie Database The Black Stallion by Walter Farley Books
The Black Stallion book series Films The Black Stallion ( 1979 ) The Black Stallion Returns ( 1983 ) The Young Black Stallion ( 2003 ) Television The Adventures
of the Black Stallion ( 1990 - 1993 ) Related Cass Ole El Mokhtar Docs Keepin Time Films directed by Carroll Ballard The Black Stallion ( 1979 ) Never Cry Wolf
( 1983 ) Nutcracker : The Motion Picture ( 1986 ) Wind ( 1992 ) Fly Away Home ( 1996 ) Duma ( 2005 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 5>

<item 6>
*The Black Stallion* is a 1979 American adventure film directed by Carroll Ballard, adapted from Walter Farley’s novel. Shipwrecked in 1946, boy Alec Ramsey
(Kelly Reno) befriends a wild Arabian stallion on a deserted island. Rescued together, they return to New York, where retired jockey Henry Dailey (Mickey
Rooney) trains them for a match race against two champions; the Black wins despite an injury. Featuring Teri Garr and
</item 6>

<item 7>
The Black Stallion (film)
In 2002 , it was selected for preservation in the United States National Film Registry by the Library of Congress as being `` culturally , historically , or
aesthetically significant '' . Also , the film is recognized by American Film Institute in these lists : 2001 : AFI 's 100 Years ... 100 Thrills -- Nominated
2005 : AFI 's 100 Years of Film Scores -- Nominated 2006 : AFI 's 100 Years ... 100 Cheers -- # 64 Sequel , prequel and television series ( edit ) The film was
followed in 1983 by a sequel , The Black Stallion Returns , which also starred Reno . There was also a television series called The Adventures of the Black
Stallion which aired from 1990 to 1993 and starred Mickey Rooney and Richard Ian Cox . In 2003 , a 50 - minute prequel called The Young Black Stallion , was
shot and released for IMAX theaters . See also ( edit ) The Black Stallion books References ( edit ) Jump up ^ `` The Black Stallion , Box Office Information ''
. The Numbers .
</item 7>

<item 8>
The Black Stallion (film)
wikipedia.org/w/index.php?title=The_Black_Stallion_(film)&oldid=836268326 '' Categories : 1979 films English - language films American children 's films Films
about horses United States National Film Registry films United Artists films American Zoetrope films Directorial debut films Films shot in Sardinia Films
directed by Carroll Ballard Films that won the Best Sound Editing Academy Award Horse racing films Screenplays by Melissa Mathison American films Films set on
islands Films set in 1946 Hidden categories : Pages using div col with deprecated parameters Talk Contents About Wikipedia Català Dansk Deutsch Español فارسی
Français Bahasa Indonesia Italiano Bahasa Melayu Nederlands 日本 語 Polski Português Русский Svenska 6 more Edit links This page was last edited on 13 April 2018 ,
at 18 : 25 . About Wikipedia
</item 8>

<item 9>
The Black Stallion (film)
After being rescued , they are set on entering a race challenging two champion horses . The film is adapted by Melissa Mathison , Jeanne Rosenberg and William
D. Wittliff . It is directed by Carroll Ballard . The movie stars Kelly Reno , Mickey Rooney , Teri Garr , Hoyt Axton , and the Arabian horse Cass Ole . The
film features music by Carmine Coppola , the father of Hollywood producer Francis Ford Coppola , who was the executive producer of the film . Contents ( hide )
1 Plot 2 Cast 2.1 Horses 3 Awards and honors 3.1 Academy Awards 3.2 Golden Globe Awards 3.3 British Academy Awards 3.4 LA Film Critics Awards 3.5 Others 4
Sequel , prequel and television series 5 See also 6 References 7 External links Plot ( edit ) In 1946 Alec Ramsey is aboard the steamer Drake off the coast of
North Africa , where he sees a wild black stallion being forced into a makeshift stable compartment and heavily restrained by ropes leading to his halter .
</item 9>

<item 10>
The Black Stallion (film)
The Black stallion ( film ) - wikipedia The Black stallion ( film ) Jump to : navigation , search The Black Stallion 1979 poster Directed by Carroll Ballard
Produced by Fred Roos Francis Ford Coppola Tom Sternberg Screenplay by Melissa Mathison Jeanne Rosenberg William D. Wittliff Based on The Black Stallion by
Walter Farley Starring Kelly Reno Mickey Rooney Teri Garr Music by Carmine Coppola Cinematography Caleb Deschanel Edited by Robert Dalva Production company
American Zoetrope Distributed by United Artists Release date October 17 , 1979 ( 1979 - 10 - 17 ) Running time 118 minutes Country United States Language
English Budget $2.7 million Box office $37.8 million The Black Stallion is a 1979 American adventure film based on the 1941 classic children 's novel The Black
Stallion by Walter Farley . But the film starts in 1946 , five years after the book was published . It tells the story of Alec Ramsey , who is shipwrecked on a
deserted island with a wild Arabian stallion whom he befriends .
</item 10>
=== END ITEM id=def2e96b1e7b767b ===

=== ITEM id=bd7b45c259574ff3 ===
Gold sentences and facts:
Sentence 0: A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called `` perfecting the security interest '' in the property , and this type of loan is a secured loan . A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit documents and the lease , respectively , and not the financing statement .
  Fact 0.0: A UCC-1 financing statement is a legal form.
  Fact 0.1: UCC-1 is an abbreviation for Uniform Commercial Code-1.
  Fact 0.2: A creditor files a UCC-1 financing statement to give notice that it has or may have an interest in the personal property of a debtor.
  Fact 0.3: A debtor is a person who owes a debt to the creditor, as typically specified in the agreement creating the debt.
  Fact 0.4: A financing statement is filed to perfect a creditor's security interest.
  Fact 0.5: The filing gives public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt, with a certain priority.
  Fact 0.6: Such notices of sale are often found in local newspapers.
  Fact 0.7: Once the form has been filed, the creditor establishes a relative priority with other creditors of the debtor.
  Fact 0.8: Perfecting the security interest is also called perfecting the security interest in the property.
  Fact 0.9: This type of loan is a secured loan.
  Fact 0.10: A financing statement may also be filed in the real estate records by a lessor of fixtures.
  Fact 0.11: The real estate filing establishes the lessor's priority against a holder of a mortgage or other lien on the real property.
  Fact 0.12: The creditor's rights against the debtor are based on the credit documents, not the financing statement.
  Fact 0.13: The lessor's rights against the lessee are based on the lease, not the financing statement.
Sentence 1: The financing statement is generally filed with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which is tasked with overseeing business organizations and activities , including receipt of financing statements . However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures . In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where third parties are most likely to search for such record .
  Fact 1.0: The financing statement is generally filed with the office of the state secretary of state.
  Fact 1.1: The financing statement is generally filed in the state where the debtor is located.
  Fact 1.2: For an individual, the state where the debtor is located is the state where the debtor resides.
  Fact 1.3: For most kinds of business organizations, the state where the debtor is located is the state of incorporation or organization.
  Fact 1.4: Many states have a state agency that operates under the secretary of state.
  Fact 1.5: That state agency is tasked with overseeing business organizations and activities, including receipt of financing statements.
  Fact 1.6: An exception exists if the collateral is tied to a particular piece of real property, such as timber, mineral rights, or fixtures.
  Fact 1.7: In that exception case, the filing must be made in the county where the property is located.
  Fact 1.8: In that exception case, the filing is usually made in the recording office or county court.
  Fact 1.9: Third parties are most likely to search for such a record in that county.

Retrieved items:
<item 1>
UCC-1 financing statement
However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures
. In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where
third parties are most likely to search for such record . In the case of a loan secured by personal property collateral , the filing of a financing statement
gives notice of a lien against the property so that other lenders or buyers of the personal property will know of the security interest . In the case of a
filing of a financing statement by a lessor of fixtures , the filing of the financing statement gives notice of the lessor 's interests to others who acquire an
interest in the real property and related fixtures . The financing statement does not create a lien nor does it create any additional rights against a lessee in
favor of a lessor , the filing of a financing statement just gives notice of whatever rights the creditor or lessor have under their loan documents or lease ,
respectively .
</item 1>

<item 2>
Ownership
A legal entity is a legal construct through which the law allows a group of natural persons to act as if it were an individual for certain purposes . Some duly
incorporated entities may not be owned by individuals nor by other entities ; they exist without being owned once they are created . Not being owned , they can
not be bought and sold . Mutual life insurance companies , credit unions , foundations and cooperatives , not for profit organizations , and public corporations
are examples of this . No person can purchase the company , as their ownership is not legally available for sale , neither as shares nor as a single whole .
Intellectual property ( edit ) Main article : Intellectual property Intellectual property ( IP ) refers to a legal entitlement which sometimes attaches to the
expressed form of an idea , or to some other intangible subject matter . This legal entitlement generally enables its holder to exercise exclusive rights of use
in relation to the subject matter of the IP . The term intellectual property reflects the idea that this subject matter is the product of the mind or the
intellect , and that IP rights may be protected at law in the same way as any other form of property .
</item 2>

<item 3>
UCC-1 financing statement
UCC - 1 financing statement - wikipedia UCC - 1 financing statement Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2014 ) ( Learn how and
when to remove this template message ) A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files
to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the
agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to
take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local
newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called ``
perfecting the security interest '' in the property , and this type of loan is a secured loan .
</item 3>

<item 4>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 4>

<item 5>
IRS penalties
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 5>

<item 6>
UCC-1 financing statement
A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of
a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit
documents and the lease , respectively , and not the financing statement . Pursuant to the standards set forth in the UCC , at 9 - 503 and 9 - 504 , the
financing statement need only contain three pieces of information : the debtor 's name and address the creditor 's name and address an indication of the
collateral , `` whether or not it is specific , if it reasonably identifies what is described . '' ( UCC 9 - 108 ) The financing statement is generally filed
with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most
kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which
is tasked with overseeing business organizations and activities , including receipt of financing statements .
</item 6>

<item 7>
Bond (finance)
situation Squeeze out Staggered board of directors Stock swap Super-majority amendment Tag - along right Takeover Reverse Tender offer Leverage Debt
restructuring Debtor - in - possession financing Financial sponsor Leveraged buyout Leveraged recapitalization High - yield debt Private equity Project finance
Valuation Accretion / dilution analysis Adjusted present value Associate company Business valuation Conglomerate discount Cost of capital Weighted average
Discounted cash flow Economic Value Added Enterprise value Fairness opinion Financial modeling Free cash flow Free cash flow to equity Market value added
Minority interest Modigliani -- Miller theorem Net present value Pure play Real options Residual income Stock valuation Sum - of - the - parts analysis Tax
shield Terminal value Valuation using multiples List of investment banks Outline of finance Debt Instruments Bonds Corporate Debenture Government Municipal
Loans Business loan Consumer lending Loan shark Payday loan Predatory lending Usury Management Bankruptcy Consolidation Management plan Relief Restructuring
Debt - snowball method Debtor - in - possession ( DIP ) financing Loan guarantee Collection Evasion Bad debt Charge - off Collection agency Compliance Debt
bondage Debtors ' prison Distraint Garnishment Phantom debt Strategic default Tax refund interception Markets Consumer Corporate Government Municipal Venture
Buyer Deposit account Fixed income Money market Securitization
</item 7>

<item 8>
UCC-1 financing statement
References ( edit ) Jump up ^ http://www.findlegalforms.com/articles/form-encyclopedia/what-is-a-ucc1-financing-statement Jump up ^
http://biztaxlaw.about.com/od/glossaryu/g/ucc1statement.htm Retrieved from ``
https://en.wikipedia.org/w/index.php?title=UCC-1_financing_statement&oldid=803697764 '' Categories : Uniform Commercial Code Legal terminology Hidden categories
: Articles needing additional references from October 2014 All articles needing additional references Talk About Wikipedia Add links This page was last edited
on 4 October 2017 , at 02 : 00 . About Wikipedia
</item 8>

<item 9>
IRS penalties
Jump up ^ 26 USC 6651 ( a ) ( 1 ) . Jump up ^ IRC § 6651 ( a ) ( 1 ) Jump up ^ IRC § 6651 ( a ) ( 2 ) Jump up ^ `` 1040 ( 2015 ) '' . Internal Revenue Service .
Retrieved June 28 , 2016 . Jump up ^ 26 USC 6662 . Jump up ^ 26 USC 6721 . Jump up ^ 26 CFR 301 - 6721 - 1 ( a ) ( 2 ) ( ii ) . Jump up ^ 26 USC 6698 . Jump up
^ 26 USC 6699 . Jump up ^ Technically , the Form 1120S is an income tax return , not an information return . Jump up ^ 26 USC 3101 - 3406 . Jump up ^ 26 USC
6672 . Jump up ^ 26 USC 3504 . Jump up ^ See , e.g. , United States v. St. Clair , 45 Amer . Fed . Tax Rep.
</item 9>

<item 10>
Ownership
A structured group duly constituted as an entity under law may still not protect members from being personally liable for each other 's actions . Court
decisions against the entity itself may give rise to unlimited personal liability for each and every member . An example of this situation is a professional
partnership ( e.g. law practice ) in some jurisdictions . Thus , being a partner or owner in a group may give little advantage in terms of share ownership while
producing a lot of risk to the partner , owner or participant . Sharing gains ( edit ) At the end of each financial year , accounting rules determine a surplus
or profit , which may be retained inside the entity or distributed among owners according to the initial setup intent when the entity was created . For public
corporations , common shareholders have no right to receive any of the profit . Entities with a member focus will give financial surplus back to members
according to the volume of financial activity that the participating member generated for the entity . Examples of this are producer cooperatives , buyer
cooperatives and participating whole life policyholders in both mutual and share - capital insurance companies . Entities with share voting rights that depend
on financial capital distribute surplus among shareholders without regard to any other contribution to the entity .
</item 10>
=== END ITEM id=bd7b45c259574ff3 ===

=== ITEM id=77d3854c6635a8bb ===
Gold sentences and facts:
Sentence 0: All black military pilots who trained in the United States trained at Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic .
  Fact 0.0: All black military pilots who trained in the United States trained at Moton Field.
  Fact 0.1: All black military pilots who trained in the United States trained at the Tuskegee Army Air Field.
  Fact 0.2: All black military pilots who trained in the United States were educated at Tuskegee University.
  Fact 0.3: Tuskegee University is located near Tuskegee, Alabama.
  Fact 0.4: The group included five Haitians from the Haitian Air Force.
  Fact 0.5: The group included one pilot from Trinidad.
  Fact 0.6: The group included a Hispanic or Latino airman born in the Dominican Republic.

Retrieved items:
<item 1>
Tuskegee Airmen
This was one of the earliest racially integrated courses in the U.S. Army . Seventeen flight surgeons served with the Tuskegee Airmen from 1941 - 49 . At that
time , the typical tour of duty for a U.S. Army flight surgeon was four years . Six of these physicians lived under field conditions during operations in North
Africa , Sicily , and Italy . The chief flight surgeon to the Tuskegee Airmen was Dr. Vance H. Marchbanks , Jr. , MD , a childhood friend of Benjamin Davis .
The accumulation of washed - out cadets at Tuskegee and the propensity of other commands to `` dump '' African - American personnel on the post exacerbated the
difficulties of administering Tuskegee . A shortage of jobs for them made these enlisted men a drag on Tuskegee 's housing and culinary departments . Trained
officers were also left idle , as the plan to shift African - American officers into command slots stalled , and white officers not only continued to hold
command , but were joined by additional white officers assigned to the post .
</item 1>

<item 2>
Tuskegee Airmen
Some ground crews trained at Mather before rotating to Inglewood . Gunners learned to shoot at Eglin Field , Florida . Bombers - navigators learned their trades
at Hondo Army Air Field and Midland Air Field , Texas or at Roswell , New Mexico . Training of the new African - American crewmen also took place at Sioux Falls
, South Dakota , Lincoln , Nebraska , and Scott Field , Belleville , Illinois . Once trained , the air and ground crews would be spliced into a working unit at
Selfridge . Command difficulties ( edit ) The new group 's first commanding officer was Colonel Robert Selway , who had also commanded the 332nd Fighter Group
before it deployed for combat overseas . Like his ranking officer , Major General Frank O'Driscoll Hunter from Georgia , Selway was a racial segregationist .
Hunter was blunt about it , saying such things as , `` ... racial friction will occur if colored and white pilots are trained together . '' He backed Selway 's
violations of Army Regulation 210 - 10 , which forbade segregation of air base facilities .
</item 2>

<item 3>
Tuskegee Airmen
This small number of enlisted men became the core of other black squadrons forming at Tuskegee and Maxwell Fields in Alabama . The Tuskegee program officially
began June 1941 with the 99th Pursuit Squadron at Tuskegee University . The unit consisted of 47 officers and 429 enlisted men , and was backed by an entire
service arm . After primary training at Moton Field , they were moved to the nearby Tuskegee Army Air Field , about 10 miles ( 16 km ) to the west for
conversion training onto operational types . Consequently , Tuskegee Army Air Field became the only Army installation performing three phases of pilot training
( basic , advanced , and transition ) at a single location . Initial planning called for 500 personnel in residence at a time . By mid-1942 , over six times
that many were stationed at Tuskegee , even though only two squadrons were training there . War poster featuring a Tuskegee Airman Tuskegee Army Airfield was
similar to already - existing airfields reserved for training white pilots , such as Maxwell Field , only 40 miles ( 64 km ) distant .
</item 3>

<item 4>
Tuskegee Airmen
Finally , on 3 April 1939 , Appropriations Bill Public Law 18 was passed by Congress containing an amendment by Senator Harry H. Schwartz , designating funds
for training African - American pilots . The War Department managed to put the money into funds of civilian flight schools willing to train black Americans .
War Department tradition and policy mandated the segregation of African - Americans into separate military units staffed by white officers , as had been done
previously with the 9th Cavalry , 10th Cavalry , 24th Infantry Regiment and 25th Infantry Regiment . When the appropriation of funds for aviation training
created opportunities for pilot cadets , their numbers diminished the rosters of these older units . In 1941 , the War Department and the Army Air Corps , under
pressure -- three months before its transformation into the USAAF -- constituted the first all - black flying unit , the 99th Pursuit Squadron . Due to the
restrictive nature of selection policies , the situation did not seem promising for African - Americans since , in 1940 , the U.S. Census Bureau reported there
were only 124 African - American pilots in the nation .
</item 4>

<item 5>
Tuskegee Airmen
They formed the 332nd Fighter Group and the 477th Bombardment Group of the United States Army Air Forces . The name also applies to the navigators , bombardiers
, mechanics , instructors , crew chiefs , nurses , cooks and other support personnel . All black military pilots who trained in the United States trained at
Moton Field , the Tuskegee Army Air Field , and were educated at Tuskegee University , located near Tuskegee , Alabama . The group included five Haitians from
the Haitian Air Force , and one pilot from Trinidad. . It also included a Hispanic or Latino airman born in the Dominican Republic . Although the 477th
Bombardment Group trained with North American B - 25 Mitchell bombers , they never served in combat . The 99th Pursuit Squadron ( later , 99th Fighter Squadron
) was the first black flying squadron , and the first to deploy overseas ( to North Africa in April 1943 , and later to Sicily and Italy ) . The 332nd Fighter
Group , which originally included the 100th , 301st , and 302nd Fighter Squadrons , was the first black flying group .
</item 5>

<item 6>
Tuskegee Airmen
The white population of Freeman Field was 250 officers and 600 enlisted men . Superimposed on it were 400 African - American officers and 2,500 enlisted men of
the 477th and its associated units . Freeman Field had a firing range , usable runways , and other amenities useful for training . African - American airmen
would work in proximity with white ones ; both would live in a public housing project adjacent to the base . Colonel Selway turned the non-commissioned officers
out of their club and turned it into a second officers club . He then classified all white personnel as cadre , and all African - Americans as trainees . One
officers club became the cadre 's club . The old Non-Commissioned Officers Club , promptly sarcastically dubbed `` Uncle Tom 's Cabin '' , became the trainee 's
officers club . At least four of the trainees had flown combat in Europe as fighter pilots , and had about four years in service . Four others had completed
training as pilots , bombardiers and navigators , and may have been the only triply qualified officers in the entire Air Corps .
</item 6>

<item 7>
Tuskegee Airmen
Census Bureau reported there were only 124 African - American pilots in the nation . The exclusionary policies failed dramatically when the Air Corps received
an abundance of applications from men who qualified , even under the restrictive requirements . Many of the applicants already had participated in the Civilian
Pilot Training Program , unveiled in late December 1938 ( CPTP ) . Tuskegee University had participated since 1939 . Testing ( edit ) This section does not cite
any sources . Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) (
Learn how and when to remove this template message ) The U.S. Army Air Corps had established the Psychological Research Unit 1 at Maxwell Army Air Field ,
Montgomery , Alabama , and other units around the country for aviation cadet training , which included the identification , selection , education , and training
of pilots , navigators , and bombardiers .
</item 7>

<item 8>
Tuskegee Airmen
Psychologists employed in these research studies and training programs used some of the first standardized tests to quantify IQ , dexterity and leadership
qualities to select and train the best - suited personnel for the roles of bombardier , navigator , and pilot . The Air Corps determined that the existing
programs would be used for all units , including all - black units . At Tuskegee , this effort continued with the selection and training of the Tuskegee Airmen
. The War Department set up a system to accept only those with a level of flight experience or higher education which ensured that only the most able and
intelligent African - American applicants were able to join . The first Lady 's flight ( edit ) The budding flight program at Tuskegee received a publicity
boost when First Lady Eleanor Roosevelt inspected it in March 1941 , and flew with African - American chief civilian instructor C. Alfred `` Chief '' Anderson .
Anderson , who had been flying since 1929 , and was responsible for training thousands of rookie pilots , took his prestigious passenger on a half - hour flight
in a Piper J - 3 Cub . After landing , she cheerfully announced , `` Well , you can fly all right . ''
</item 8>

<item 9>
Tuskegee Airmen
Several of the Tuskegee Airmen had logged over 900 flight hours by this time . Nevertheless , by Colonel Selway 's fiat , they were trainees . Off - base was no
better ; many businesses in Seymour would not serve African - Americans . A local laundry would not wash their clothes , yet willingly laundered those of
captured German soldiers . In early April 1945 , the 118th Base Unit transferred in from Godman Field ; its African - American personnel held orders that
specified they were base cadre , not trainees . On 5 April , officers of the 477th peaceably tried to enter the whites - only Officer 's Club . Selway had been
tipped off by a phone call , and had the assistant provost marshal and base billeting manager stationed at the door to refuse the 477th officers entry . The
latter , a major , ordered them to leave , and took their names as a means of arresting them when they refused . It was the beginning of the Freeman Field
Mutiny .
</item 9>

<item 10>
Tuskegee Airmen
Military Red Ball Express Strategic bombing during World War II Port Chicago disaster Walterboro Army Airfield training site and memorial Willa Brown Notes (
edit ) Jump up ^ U.S. state and local laws enacted between 1876 and 1965 that mandated de jure racial segregation in all public activities were collectively
known as the `` Jim Crow laws '' ; the name derivation from a contemporary pejorative for Black Americans . Jump up ^ `` Pursuit '' being the U.S. term for ``
fighter '' to May 1942 . Jump up ^ It may have been a lawsuit from a rejected candidate , that caused the USAAC to accept black applicants . Yancey Williams ,
an African American otherwise qualified for military pilot 's training , filed suit in the District Court in Washington , DC for admittance to training . He was
backed by the National Association of for the Advancement of Colored People . Jump up ^ The Coffey School of Aeronautics in Chicago was also considered .
Tuskegee 's already existent CPTP , ground school training facilities , availability of nearby Montgomery Field , and better flying weather certainly offered
some advantages .
</item 10>
=== END ITEM id=77d3854c6635a8bb ===

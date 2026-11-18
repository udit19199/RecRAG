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

=== ITEM id=9dbd468f54219652 ===
Gold sentences and facts:
Sentence 0: Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 % WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X 10.0 -- 10.1. 5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS / 2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS. TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts
  Fact 0.0: The hosts file on Unix, Unix-like, and POSIX systems is at /etc/hosts
  Fact 0.1: On Microsoft Windows 3.1 the hosts file is at %WinDir%\HOSTS
  Fact 0.2: On Windows 95, 98, and ME the hosts file is at %WinDir%\hosts
  Fact 0.3: On Windows NT, 2000, XP, 2003, Vista, 2008, 7, 2012, 8, and 10 the hosts file is at %SystemRoot%\System32\drivers\etc\hosts
  Fact 0.4: On Windows Mobile and Windows Phone the hosts entry is a registry key under HKEY_LOCAL_MACHINE\Comm\Tcpip\Hosts
  Fact 0.5: On Apple Macintosh 9 and earlier the hosts file is in the Preferences or System folder
  Fact 0.6: On Mac OS X 10.0 to 10.1.5 the hosts file was added through NetInfo or niload
  Fact 0.7: On Mac OS X 10.2 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.8: On Novell NetWare the hosts file is at SYS:etc\hosts
  Fact 0.9: On OS/2 and eComStation the hosts file is at the bootdrive path \mptn\etc\
  Fact 0.10: On Symbian OS 6.1 to 9.0 the hosts file is at C:\system\data\hosts
  Fact 0.11: On Symbian OS 9.1 and later the hosts file is at C:\private\10000882\hosts
  Fact 0.12: On MorphOS (NetStack) the hosts file is at ENVARC:sys/net/hosts
  Fact 0.13: On AmigaOS before 4 the hosts file is at AmiTCP:db/hosts or DEVS:Internet/hosts
  Fact 0.14: On AROS the hosts file is at ENVARC:AROSTCP/db/hosts
  Fact 0.15: On Android the hosts file is /etc/hosts, a symbolic link to /system/etc/hosts
  Fact 0.16: On iOS 2.0 and newer the hosts file is /etc/hosts, a symbolic link to /private/etc/hosts
  Fact 0.17: On TOPS-20 the hosts file is <SYSTEM>HOSTS.TXT
  Fact 0.18: On Plan 9 the hosts file is /lib/ndb/hosts
  Fact 0.19: On BeOS the hosts file is /boot/beos/etc/hosts
  Fact 0.20: On Haiku the hosts file is /boot/common/settings/network/hosts
  Fact 0.21: On OpenVMS with UCX the hosts file is UCX$HOST
  Fact 0.22: On OpenVMS with TCPware the hosts file is TCPIP$HOST
  Fact 0.23: On RISC OS 3.7 and 5 the hosts file is at !Boot.Resources.!Internet.files.Hosts, loaded in the later boot sequence
  Fact 0.24: On RISC OS the disabled hosts file is at !Boot.Choices.Hardware.Disabled.Internet.Files.Hosts

Retrieved items:
<item 1>
hosts (file)
It is a common part of an operating system 's Internet Protocol ( IP ) implementation , and serves the function of translating human - friendly hostnames into
numeric protocol addresses , called IP addresses , that identify and locate a host in an IP network . In some operating systems , the contents of the hosts file
is used preferentially to other name resolution methods , such as the Domain Name System ( DNS ) , but many systems implement name service switches , e.g. ,
nsswitch. conf for Linux and Unix , to provide customization . Unlike remote DNS resolvers , the hosts file is under the direct control of the local computer 's
administrator . File content ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed . ( April 2017 ) ( Learn how and when to remove this template message ) The hosts file
contains lines of text consisting of an IP address in the first text field followed by one or more host names . Each field is separated by white space -- tabs
are often preferred for historical reasons , but spaces are also used .
</item 1>

<item 2>
hosts (file)
It is usually named hosts , without an extension . Operating System Version ( s ) Location Unix , Unix - like , POSIX / etc / hosts Microsoft Windows 3.1 %
WinDir % \ HOSTS 95 , 98 , ME % WinDir % \ hosts NT , 2000 , XP , 2003 , Vista , 2008 , 7 , 2012 , 8 , 10 % SystemRoot % \ System32 \ drivers \ etc \ hosts
Windows Mobile , Windows Phone Registry key under HKEY_LOCAL_MACHINE \ Comm \ Tcpip \ Hosts Apple Macintosh 9 and earlier Preferences or System folder Mac OS X
10.0 -- 10.1.
</item 2>

<item 3>
hosts (file)
Hosts ( file ) - wikipedia Hosts ( file ) Jump to : navigation , search For other uses , see Host ( disambiguation ) . The computer file hosts is an operating
system file that maps hostnames to IP addresses . It is a plain text file . Originally a file named HOSTS. TXT was manually maintained and made available via
file sharing by Stanford Research Institute for the ARPANET membership , containing the hostnames and address of hosts as contributed for inclusion by member
organizations . The Domain Name System , first described in 1983 and implemented in 1984 , automated the publication process and provided instantaneous and
dynamic hostname resolution in the rapidly growing network . In modern operating systems , the hosts file remains an alternative name resolution mechanism ,
configurable often as part of facilities such as the Name Service Switch as either the primary method or as a fallback method . Contents ( hide ) 1 Purpose 2
File content 3 Location in the file system 4 History 5 Extended applications 6 Security issues 7 See also 8 References 9 External links Purpose ( edit ) The
hosts file is one of several system facilities that assists in addressing network nodes in a computer network .
</item 3>

<item 4>
hosts (file)
Network nodes typically had one address and could have many names . As local area TCP / IP computer networks gained popularity , however , the maintenance of
hosts files became a larger burden on system administrators as networks and network nodes were being added to the system with increasing frequency .
Standardization efforts , such as the format specification of the file HOSTS. TXT in RFC 952 , and distribution protocols , e.g. , the hostname server described
in RFC 953 , helped with these problems , but the centralized and monolithic nature of hosts files eventually necessitated the creation of the distributed
Domain Name System ( DNS ) . On some old systems a file named networks is present that has similar to hosts file functions containing names of networks .
Extended applications ( edit ) In its function of resolving host names , the hosts file may be used to define any hostname or domain name for use in the local
system . Redirecting local domains Some web service and intranet developers and administrators define locally defined domains in a LAN for various purposes ,
such as accessing the company 's internal resources or to test local websites in development .
</item 4>

<item 5>
hosts (file)
TXT Plan 9 / lib / ndb / hosts BeOS / boot / beos / etc / hosts Haiku / boot / common / settings / network / hosts OpenVMS UCX UCX $ HOST TCPware TCPIP $ HOST
RISC OS 3.7 , 5 ! Boot. Resources. ! Internet. files. Hosts later boot sequence ! Boot. Choices. Hardware. Disabled. Internet. Files. Hosts History ( edit ) The
ARPANET , the predecessor of the Internet , had no distributed host name database . Each network node maintained its own map of the network nodes as needed and
assigned them names that were memorable to the users of the system . There was no method for ensuring that all references to a given node in a network were
using the same name , nor was there a way to read the hosts file of another computer to automatically obtain a copy . The small size of the ARPANET kept the
administrative overhead small to maintain an accurate hosts file . Network nodes typically had one address and could have many names .
</item 5>

<item 6>
hosts (file)
5 ( Added through NetInfo or niload ) Mac OS X 10.2 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) Novell NetWare SYS : etc \ hosts OS /
2 & eComStation `` bootdrive '' : \ mptn \ etc \ Symbian Symbian OS 6.1 -- 9.0 C : \ system \ data \ hosts Symbian OS 9.1 + C : \ private \ 10000882 \ hosts
MorphOS NetStack ENVARC : sys / net / hosts AmigaOS < 4 AmiTCP : db / hosts DEVS : Internet / hosts AROS ENVARC : AROSTCP / db / hosts Android / etc / hosts ( a
symbolic link to / system / etc / hosts ) iOS iOS 2.0 and newer / etc / hosts ( a symbolic link to / private / etc / hosts ) TOPS - 20 < SYSTEM > HOSTS.
</item 6>

<item 7>
hosts (file)
Retrieved November 30 , 2012 . Jump up ^ RISC OS 6.14 Jump up ^ `` Remove Trojan. Qhosts -- Symantec '' . Retrieved May 16 , 2010 . External links ( edit )
Windows TCP / IP in Windows 2000 Professional Microsoft TCP / IP Host Name Resolution Order Blocking Unwanted Connections with a Hosts File ( Creative Commons
License ) Solution if Windows stops you from modifying the Hosts file or the Lmhosts file in Windows Vista / 7 / 8 Beginners Guide on Editing Hosts File List of
Microsoft related domains that Windows wo n't redirect Mac How to block unwanted domains with a HOSTS file on Mac OS X 10.2 or later `` Gas Mask '' - Hosts file
manager for Mac OS X Linux Block unwanted advertisements with / etc / hosts file on Linux Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
hosts (file)
Comment lines may be included ; they are indicated by an octothorpe ( # ) in the first position of such lines . Entirely blank lines in the file are ignored .
For example , a typical hosts file may contain the following : 127.0. 0.1 localhost loopback : : 1 localhost This example only contains entries for the loopback
addresses of the system and their host names , a typical default content of the hosts file . The example illustrates that an IP address may have multiple host
names ( localhost and loopback ) , and that a host name may be mapped to both IPv4 and IPv6 IP addresses , as shown on the first and second lines respectively .
Location in the file system ( edit ) The location of the hosts file in the file system hierarchy varies by operating system . It is usually named hosts ,
without an extension .
</item 8>

<item 9>
hosts (file)
History of BIND . Retrieved 2017 - 07 - 01 . Jump up ^ `` Cisco Networking Academy Program : First - Year Companion Guide '' , Cisco Systems , Inc. , 2002 ( 2nd
Edition ) , page 676 , ISBN 1 - 58713 - 025 - 4 Jump up ^ `` Linux Network Administrators Guide : Writing hosts and networks files '' . Retrieved May 16 , 2010
. Jump up ^ `` Hosts File '' . Retrieved August 10 , 2011 . Jump up ^ `` Microsoft KB Q314053 : TCP / IP and NBT configuration parameters for Windows XP '' .
Retrieved August 28 , 2010 . Jump up ^ `` Microsoft KB 972034 Revision 2.0 : default hosts files '' . Retrieved August 28 , 2010 . ^ Jump up to : `` Mac OS X :
How to Add Hosts to Local Hosts File '' . Retrieved August 28 , 2010 . ^ Jump up to : `` The Haiku / BeOS Tip Server '' . Retrieved November 30 , 2012 .
</item 9>

<item 10>
hosts (file)
Internet resource blocking Entries in the hosts file may be used to block online advertising , or the domains of known malicious resources and servers that
contain spyware , adware , and other malware . This may be achieved by adding entries for those sites to redirect requests to another address that does not
exist or to a harmless destination . Commercial software applications may be used to populate the hosts file with entries of known undesirable Internet
resources automatically . Security issues ( edit ) The hosts file may present an attack vector for malicious software . The file may be modified , for example ,
by adware , computer viruses , or trojan horse software to redirect traffic from the intended destination to sites hosting malicious or unwanted content . The
widespread computer worm Mydoom. B blocked users from visiting sites about computer security and antivirus software and also affected access from the
compromised computer to the Microsoft Windows Update website . See also ( edit ) DNSBL , a DNS - based blackhole list Content - control software Ad filtering
References ( edit ) Jump up ^ Internet Systems Consortium . `` The Most Widely Used Name Server Software : BIND '' . History of BIND . Retrieved 2017 - 07 - 01
.
</item 10>
=== END ITEM id=9dbd468f54219652 ===

=== ITEM id=78ebfc3a3831b6fe ===
Gold sentences and facts:
Sentence 0: Nintendo claims that the design of the Pro Controller is an `` enhanced version '' of the Classic Controller and `` offers a richer experience '' . Certain games with flexible control schemes , such as Call of Duty : Black Ops II and Trine 2 : Director 's Cut , are also compatible with the Classic Controller . The Wii U Pro Controller is not compatible with previous - generation Wii games .
  Fact 0.0: Nintendo claims that the design of the Wii U Pro Controller is an enhanced version of the Classic Controller.
  Fact 0.1: Nintendo claims that the Wii U Pro Controller offers a richer experience.
  Fact 0.2: Certain games with flexible control schemes are compatible with the Classic Controller.
  Fact 0.3: Call of Duty: Black Ops II is a game with a flexible control scheme that is compatible with the Classic Controller.
  Fact 0.4: Trine 2: Director's Cut is a game with a flexible control scheme that is compatible with the Classic Controller.
  Fact 0.5: The Wii U Pro Controller is not compatible with previous-generation Wii games.
Sentence 1: The Wii U Pro Controller ( Wii U PRO コントローラー ) is a video game controller produced by Nintendo for the Wii U video game console .
  Fact 1.0: The Wii U Pro Controller is a video game controller.
  Fact 1.1: The Wii U Pro Controller is produced by Nintendo.
  Fact 1.2: The Wii U Pro Controller is made for the Wii U video game console.
Sentence 2: The controller functions as a secondary controller released for the console , available separately . The Wii U system can be connected to up to four Wii U Pro Controllers at one time , however Super Smash Bros. for Wii U allows the usage of more than 4 . Like more traditional controllers , it features standard analog sticks and face buttons . Like the Wii U GamePad and Classic Controller Pro and unlike other eighth generation controllers ( i.e. , the DualShock 4 and the Xbox One Controller ) , the triggers are digital ( i.e. , not analog ) .
  Fact 2.0: The Wii U Pro Controller functions as a secondary controller for the Wii U.
  Fact 2.1: The Wii U Pro Controller was released for the console.
  Fact 2.2: The Wii U Pro Controller is available separately.
  Fact 2.3: The Wii U console can be connected to up to four Wii U Pro Controllers at one time.
  Fact 2.4: Super Smash Bros. for Wii U allows the use of more than four controllers.
  Fact 2.5: The Wii U Pro Controller has standard analog sticks.
  Fact 2.6: The Wii U Pro Controller has face buttons, like more traditional controllers.
  Fact 2.7: The Wii U Pro Controller has digital triggers, not analog triggers.
  Fact 2.8: The Wii U GamePad has digital triggers.
  Fact 2.9: The Classic Controller Pro has digital triggers.
  Fact 2.10: Unlike the DualShock 4 and the Xbox One Controller, the Wii U Pro Controller has digital triggers.

Retrieved items:
<item 1>
Wii U Pro Controller
wikipedia.org/w/index.php?title=Wii_U_Pro_Controller&oldid=837880962 '' Categories : Nintendo controllers Wii U Hidden categories : Pages using deprecated image
syntax Articles containing Japanese - language text Talk About Wikipedia Deutsch Español Italiano Português Edit links This page was last edited on 23 April
2018 , at 16 : 07 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
Wii U Pro Controller
nintendolife.com . October 3 , 2013 . Retrieved August 16 , 2014 . Jump up ^ `` Wii U Pro Controller Boasts 80 - Hour Battery Life - IGN '' . Uk.ign.com .
October 16 , 2012 . Retrieved September 21 , 2013 . Jump up ^ Liebl , Matt ( June 12 , 2012 ) . `` Nintendo defends Wii U Pro Controller design '' . GameZone .
Retrieved July 28 , 2012 . Jump up ^ `` Wii U Pro Controller Not Compatible With Wii Games '' . Tomshardware.com . Retrieved November 26 , 2012 .
</item 2>

<item 3>
Wii U Pro Controller
The Wii U Pro Controller is not compatible with previous - generation Wii games . References ( edit ) Jump up ^ Franich , Darren ( June 3 , 2012 ) . `` The Wii
U Pro Controller looks very familiar ... '' EW.com . Retrieved July 28 , 2012 . Jump up ^ Kain , Erik ( June 5 , 2012 ) . `` Nintendo 's Smart Decision With Its
Wii U Pro Controller '' . Forbes . Retrieved July 28 , 2012 . Jump up ^ Grubb , Jeffrey ( June 3 , 2012 ) . `` Nintendo 's Wii U Pro Controller may make
Microsoft blush '' . VentureBeat . Retrieved July 28 , 2012 . Jump up ^ How to Sync a Wii Remote with a Wii U Console Jump up ^ `` Wii U Pro Controller and 3DS
share the same battery '' . gonintendo.com . November 9 , 2012 . Retrieved August 16 , 2014 . Jump up ^ `` We Opened Up The Nintendo 2DS And Found A 1300mAh
Battery Inside '' .
</item 3>

<item 4>
Wii U Pro Controller
Wii U Pro Controller - wikipedia Wii U Pro Controller Jump to : navigation , search Wii U Pro Controller Wii U Pro Controller Manufacturer Nintendo Type Video
game controller Generation Eighth generation Retail availability November 18 , 2012 Input 2 × analog sticks Digital d - pad 10 × digital face buttons ( two
buttons found under analog sticks ; controller features additional button for syncing purposes ) 2 × digital shoulder buttons 2 × digital triggers Connectivity
Bluetooth Power 1300 mAh , Mini USB B connector ( recharge ) Predecessor Wii Remote ( Plus ) and Classic Controller ( Pro ) Successor Nintendo Switch Pro
Controller The Wii U Pro Controller ( Wii U PRO コントローラー ) is a video game controller produced by Nintendo for the Wii U video game console . History ( edit )
Nintendo unveiled the Wii U Pro Controller at E3 2012 ; the controller is aimed towards attracting `` hardcore '' players to the platform with a more
traditional design , to make the Wii U more competitive with Sony 's and Microsoft 's offerings . Many video game journalists have noted the similarity between
the controller and Microsoft 's Xbox 360 Controller .
</item 4>

<item 5>
History of Nintendo
Law360.com . Retrieved 17 December 2014 . Jump up ^ `` Wii European launch details announced '' . GamesIndustry.biz. 15 September 2006 . Retrieved 18 December
2014 . Jump up ^ Internet Archive Wayback Machine Jump up ^ Gaudiosi , John ( 25 April 2007 ) . `` The untold story of how the Wii beat the Xbox , PlayStation
'' . CNN.com . Retrieved 18 December 2014 . Jump up ^ Stan Schroeder , Mashable ( 25 April 2011 ) . `` Nintendo confirms Wii successor for 2012 '' . CNN.com .
Retrieved 2012 - 10 - 31 . Jump up ^ `` Wii U confirmed for Europe this year '' . Metro . Jump up ^ `` WiiU Price and Release Date Announced '' . WiiU News .
2012 - 09 - 13 . Retrieved 2012 - 09 - 13 . Jump up ^ `` Nintendo Network @ E3 2011 - Introducing Wii U '' . Web.archive.org. 2011 - 06 - 15 .
</item 5>

<item 6>
History of Nintendo
Nintendo 3DS , Wii U , and Nintendo Switch ( edit ) Main articles : Nintendo 3DS , Nintendo 3DS XL , Nintendo 2DS , New Nintendo 3DS , Wii U , and Nintendo
Switch The Wii U console with Wii U GamePad On February 26 , 2011 , the Nintendo 3DS was released in Japan . It was released on March 25 and 27 to Europe and
the US , respectively . The 3DS was part of a large wave of 3D technologies being released at the time ; unlike most of these , the 3DS did not use 3D glasses
for its 3D effect , and instead used parallax barrier autostereoscopy . On April 25 , 2011 , Nintendo confirmed that they were making a new console to be
released in 2012 . On June 7 , 2011 , the console was announced as being named the Wii U . The Wii U is Nintendo 's first console to be able to play video games
in 1080p HD .
</item 6>

<item 7>
Wii U Pro Controller
Tomshardware.com . Retrieved November 26 , 2012 . Wii U Hardware Wii U GamePad Off - TV Play Wii U Pro Controller Wii accessories Wii Remote ( Plus ) MotionPlus
Nunchuk Classic Controller Balance Board Wii Speak Third - party accessories Amiibo Online service Nintendo Network Miiverse Nintendo eShop Nintendo TVii Amazon
Instant Video / LoveFilm Hulu Plus Netflix TiVo SpotPass System software Home Menu Amazon Instant Video LoveFilm Crunchyroll Hulu Napster Netflix Niconico Mii
Maker YouTube Wii Mode Wii U Chat Games List Best - selling Balance Board Pro Controller Miis Nintendo Network Wii MotionPlus Virtual Console Japan North
America PAL region Backward compatibility Wii games Wi - Fi Connection Wii applications WiiWare games North America PAL region Wii Mode Virtual Console Japan
North America PAL region Wii series Street U Fit U Party U Sports Club Miscellaneous List of Wii games on Wii U eShop Licensed toolkits Integrated Nintendo Web
Framework Unity Supported Autodesk Gameware CryEngine ( 4th generation ) Havok Gamebryo Umbra Unreal Engine 3 ← Predecessor : Wii Follow - up : Nintendo Switch
→ Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
History of Nintendo
During the Electronics Entertainment Expo in 2011 , Nintendo 's press conference put a great amount of stress on the abilities and innovations of the new
controller , a unit resembling a tablet . Many gamers were confused after the press conference about whether the Wii U is an entirely new system or just a new
controller for the Wii . The Wii U was , however , a whole new home console system developed by Nintendo . The system was released on November 18 , 2012 in
North America , November 30 , 2012 in Europe and Australia , and on December 8 , 2012 in Japan . It featured a new controller with a 6.2 inch touchscreen , two
thumb pads , four shoulder buttons , a directional pad , and four lettered buttons . It also featured an accelerometer , gyroscope , camera , Near - field
communications ( NFC ) , and built - in microphone , similar those in the Nintendo 3DS . The Nintendo Switch Console in the Dock with Red and Blue Joy - Con
controllers . On July 11 , 2015 , Satoru Iwata died from a bile duct tumor at the age of 55 .
</item 8>

<item 9>
Wii U Pro Controller
Features ( edit ) See also : List of Wii U games that use the Wii U Pro Controller The controller functions as a secondary controller released for the console ,
available separately . The Wii U system can be connected to up to four Wii U Pro Controllers at one time , however Super Smash Bros. for Wii U allows the usage
of more than 4 . Like more traditional controllers , it features standard analog sticks and face buttons . Like the Wii U GamePad and Classic Controller Pro and
unlike other eighth generation controllers ( i.e. , the DualShock 4 and the Xbox One Controller ) , the triggers are digital ( i.e. , not analog ) . The Wii U
Pro Controller uses the same 1300mAh CTR - 003 battery found in the Nintendo 3DS and 2DS , which can last up to 80 hours before needing to be recharged .
Nintendo claims that the design of the Pro Controller is an `` enhanced version '' of the Classic Controller and `` offers a richer experience '' . Certain
games with flexible control schemes , such as Call of Duty : Black Ops II and Trine 2 : Director 's Cut , are also compatible with the Classic Controller .
</item 9>

<item 10>
History of Nintendo
It was launched in Japan on March 2 , 2006 , and three months later in North America and Europe on June 11 and June 23 , 2006 , respectively . On May 25 , 2006
, Reggie Fils - Aimé was promoted to President and COO of Nintendo of America , Inc . The former President , Tatsumi Kimishima , was promoted to Chairman of the
Board and CEO . On July 7 , 2006 , Nintendo officially established a South Korean subsidiary , Nintendo Korea , in the country 's capital , Seoul , which
replaced Daewon Media as the official distributor of Nintendo products in South Korea . Original Wii console with Wii Remote In early August 2006 , it was
revealed that Nintendo , along with Microsoft , was made the target of a patent - infringement lawsuit . Leveled by the Anascape Ltd. , the suit claimed that
Nintendo 's use of analog technology in their game controllers constitutes a violation of their patents . The lawsuit sought to recover damages from both
corporations and possibly force them to stop selling controllers with the violating technology .
</item 10>
=== END ITEM id=78ebfc3a3831b6fe ===

=== ITEM id=59a2edf0859c48d7 ===
Gold sentences and facts:
Sentence 0: Yadava ( Golla )
  Fact 0.0: The text lists Golla in parentheses after Yadava.

Retrieved items:
<item 1>
Andhra Pradesh Legislative Assembly election, 2014
Palakollu Nimmala Rama Naidu 177 Narasapuram Bandaru Madhava Naidu 178 Bhimavaram Pulaparthy Ramanjaneyulu 179 Undi Vetukuri Shivaramaraju 180 Tanuku Arimilli
Radhakrishna 181 Tadepalligudem P Manikyala Rao BJP 182 Unguturu Ganni Veeranjaneyulu TDP 183 Denduluru Chintamaneni Prabhakar 184 Eluru Badeti Bujji 185
Gopalapuram ( SC ) Muppidi Venkateshwara Rao 186 Polavaram ( ST ) Modiyam Srinivasa Rao 187 Chintalapudi ( SC ) Peethala Sujatha Krishna 188 Tiruvuru ( SC )
Kokkilagadda Rakshana Nidhi YSRCP 189 Nuzvid Meka Pratap Apparao 190 Gannavaram Vallabhaneni Vamsimohan TDP 191 Gudivada Kodali Sri Venkateswara Rao YSRCP 192
Kaikalur Kamineni
</item 1>

<item 2>
Andhra Pradesh Legislative Assembly election, 2014
( SC ) P David raju 222 Darsi Shiddha Raghava Rao TDP 223 Parchur Y SAMBHA SIVA 224 Addanki G Ravi Kumar YSRCP 225 Chirala A Krishnamohan NP 226
Santhanuthalapadu ( SC ) Adimulapau Suresh YSRCP 227 Ongole Damacharla Janardhana Rao TDP 228 Kandukur Pothula Ramarao YSRCP 229 Kondapi ( SC ) DBV Swamy TDP
230 Markapuram Janke Venkata Reddy YSRCP 231 Giddalur Muttumula Ashok Reddy 232 Kanigiri Kadiri Babu rao TDP Nellore 233 Kavali PRATHAP KUMAR REDDY YSRCP 234
Atmakur Mekapati Gautam Reddy 235 Kovur Polamreddy srinivasulu reddy TDP 236 Nellore City P Anil Kumar Yadav YSRCP 237 Nellore Rural K Srihar Reddy 238
Sarvepalli Kakani
</item 2>

<item 3>
Andhra Pradesh Legislative Assembly election, 2014
Kodali Sri Venkateswara Rao YSRCP 192 Kaikalur Kamineni Srinivasa Rao BJP 193 Pedana Kagitha Venkat Rao TDP 194 Machilipatnam Kollu Ravindra 195 Avanigadda
Mandali Buddha Prasad 196 Pamarru ( SC ) Uppuleti Kalpana YSRCP 197 Penamaluru Bodepudi Prasad TDP 198 Vijayawada West Jallel Khan TDP 199 Vijayawada Central
Bonda Umamaheswara Rao TDP 200 Vijayawada East Gadde Ramamohan 201 Mylavaram Devineni Uma Maheswara Rao 202 Nandigama ( SC ) Thangirala Sowmya 203 Jaggayyapeta
Sreeram Rajagopal ( Tataiah ) Guntur 204 Pedakurapadu Sridhar 205 Tadikonda ( SC ) Tenali Sravana Kumar 206 Mangalagiri Aalla Ramakrishna Reddy YSRCP 207
Ponnuru Dhulipalla Narendra Kumar
</item 3>

<item 4>
List of Telugu castes
Mutrasi Tenugollu Munnurukapu ( Telangana ) Nagavasam ( Nagavamsa ) Neeli ( Nelli ) Pollinativelama ( of Srikakulam & Visakhapatnam Districts ) Poosala Passi
Rangarez or Bhavasara Kshatriya Sadhuchetty Satani ( Chattadasrivaishnava , Chatadi ) Tammali Turupu Kapu and Gajula Kapu of Srikakulam , Vizianagram &
Visakhapatnam Districts who are subject to social customs of divorce & remarriages among their women Uppara or Sagara Vanjara ( Vanjari ) Yadava ( Golla ) Pala
- Ekari Ayyaraka Nagaralu Pondara Lodh , Lodha , Lodhi Kurakula Kasi kapadi Gajula Balija ( who are traditionally associated with selling of Bangles ) Vannia
Vanniar Vannikula - Kshatriya Sarollu Atirasa ( of Polavaram , Gopalapuram , Koyyalagudem , Buttayagudem ,
</item 4>

<item 5>
List of Telugu castes
Puragiri Kshatriya ) Nessi or Kurni Padmasali ( Sali , Saliyan , Pattusali , Senapathulu , Thogata Sali ) Srisayana ( Segidi ) Swakulasali Thogata , Thogati or
Thogata Veerakshatriya Viswabrahmin or Viswakarma ( Ausula or Kamsali , Kammari , Kanchari , Vadla or Vadra or Vadrangi and Silpi ) Scheduled Castes converts to
Christianity and their progeny Agaru Arekatika , Katika , Quresh ( Muslim Butchers ) Atagara Bhataraju Chippolu ( Mera ) Gavara Godaba Hatkar Jakkala Jingar
Kandra Koshti Kachi Surya Balija ( Kalavanthula ) , Ganika Krishnabhalija ( Dasari , Bukka , Bukka Ayavar ) Koppulavelama Mathura Mali ( where they are not
Scheduled Tribe ) Mudiraj ,
</item 5>

<item 6>
List of Telugu castes
List of telugu castes - wikipedia List of telugu castes Jump to : navigation , search Telugu society is nowadays officially classified into a number of groups ,
some of which align with the concept of caste . Contents ( hide ) 1 Forward Classes 2 Backward Classes 3 See also 4 References Forward Classes Forward castes do
not qualify for government reservations . Those present in Andhra Pradesh are : Brahmin Kamma Kapu Komati Kshatriya Reddy Velama Some communities within these
groups are instead classified under other categories , as noted below . Backward Classes Some communities are classified as Other Backward Classes due to social
, financial , educational , and / or political discrimination . Agnikulakshatriya Palli Vadabalija Bestha Jalari Gangavar Gangaputra Goondla Vanyakulakshtriya
Vannekapu Vannereddi Pallikapu Pallireddi Nayyala Pattapu Balasanthu , Bahurupi Bandara Budabukkala Rajaka , Chakali ,
</item 6>

<item 7>
List of Telugu castes
Gopalapuram , Koyyalagudem , Buttayagudem , Chagallu Mandals of West Godavari Distt. and Devipattanam , Korukonda and Gokavaram Mandals of East Godavari
District ) Patra Sikligar Siddula Gudia / Gudiya ( confined to Srikakulam , Vizianagaram and Visakhapatnam Districts only ) Kurmi ( confined to Telangana Region
and also Krishna District only ) Budubunjala / Bhunjwa / Bhadbhunja ( confined to Hyderabad and Ranga Reddy Districts only ) Lakkamari Kapu ( confined to
Telangana Region only ) See also Scheduled Castes and Scheduled Tribes List of Scheduled Tribes in India # Andhra Pradesh References Jump up ^ Srinivasulu , K.
`` Caste , Class and Social Articulation in Andhra Pradesh : Mapping Differential Regional Trajectories '' ( PDF ) . Overseas Development Institute . p. 12 .
</item 7>

<item 8>
List of Telugu castes
Buragana Kalinga , Buragam Kalinga , Pandiri Kalinga , Kalinga Kepmare or Reddika Mondepatta Mondipatta Nokkar Pariki Muggula Yata Chopemari Kaikadi
Joshinandiwala Odde Oddilu Vaddi Vaddelu Mandula Mehtar ( Muslim ) Kunapuli Achukatlavandlu Aryakshatriya , Chittari , Giniyar , Chitrakara , Nakhas Devanga
Ediga Gowda ( Gamalla , Kalalee ) Goundla Settibalija ( of Visakhapatnam , East Godavari , West Godavari and Krishna District ) Dudekula , Laddaf , Pinjari or
Noorbash Gandla , Telikula , Devathilakula Jandra Kummara or Kulala or Salivahana Karikalabhakthulu Kaikolan or Kaikola ( Sengundam or Sengunther )
Karnabhakthulu Kuruba or Kuruma Nagavaddilu Neelakanthi Patkar ( Khatri ) Perika ( Perike Balija ,
</item 8>

<item 9>
List of Telugu castes
Bahurupi Bandara Budabukkala Rajaka , Chakali , Vannar Dasari ( formerly engaged in Bhikshatana ) Dommara Gangiredlavaru Jangam whose traditional occupation is
begging Jogi Katipapala Korcha , Koracha Medari or Mahendra Mondivaru Mondibanda Banda Nayi - Brahmin ( Mangali , Mangala , Bhajanthri ) Vamsha Raj Pamula
Pardhi ( Mirshikari , Nirshikari ) Pambala Peddammavandlu , Devaravandlu , Yellammavandlu , Mutyalammavandlu Veeramushti ( Neetikotala ) , Veerabhadreeya
Valmiki Boya ( Boya Bedar , Kirataka Nishadi , Yellappi , Yellapu / Yellapondlu , Pedda Boya ) Talayari , Chunduvallu Gudala Kanjara - Bhatta Kinthala Kalinga ,
Buragana Kalinga , Buragam Kalinga , Pandiri Kalinga ,
</item 9>

<item 10>
Andhra Pradesh Legislative Assembly election, 2014
Prabhakar Chowdary 273 Kalyandurg Hanumatharaya Chowdari 274 Raptadu Paritala Sunitha 275 Madakasira K Eeranna 276 Hindupur Nandamuri Balakrishna 277 Penukonda
BK Partha Sarathi 278 Puttaparthi Palle Raghunatha Reddy 279 Dharmavaram Suryanarayana 280 Kadiri Chand Basha Chittoor 281 Thamballapalle G. Shankar Yadav TDP
282 Pileru Chinthala Ramachandra Reddy YSRCP 283 Madanapalle D Thippa Reddy 284 Punganur P Ramachandra Reddy 285 Chandragiri C Bhaskar Reddy 286 Tirupati
Mannuru Suguna TDP 287 Srikalahasti Bojjala Gopala Krishna Reddy 288 Satyavedu Talari Aditya 289 Nagari R.K.
</item 10>
=== END ITEM id=59a2edf0859c48d7 ===

=== ITEM id=f1fc59d38caa3305 ===
Gold sentences and facts:
Sentence 0: No . Name Province Birth / death Term of active service Term of service as Chief Length in days of active service Length in days as Chief Appointed by Law school Position prior to appointment Codrington Edmund Carrington England 1769 - 1849 March 1801 -- 2 April 1806 March 1801 -- 2 April 1806 North -- Counsel to the East India Company Edmund Henry Lushington England 1766 - 1839 5 September 1801 -- 1809 15 April 1807 - 1809 North University of Cambridge Barrister Alexander Johnston Scotland 1775 - 1849 15 April 1807 -- 14 February 1809 6 November 1811 - 1819 6 November 1811 - 1819 671 + ? Maitland University of Göttingen Advocate Fiscal of Ceylon William Coke England 1776 - 1818 6 March 1809 -- 1 September 1818 † 6 March 1809 -- 1811 3,466 Maitland University of Oxford Advocate Fiscal of Ceylon 5 Henry Byrne 5 November 1818 -- 1819 -- -- Brownrigg Barrister 6 Ambrose Hardinge Giffard Ireland 1771 -- 1827 8 April 1819 -- 2 March 1827 8 April 1819 -- 2 March 1827 2,885 2,885 Brownrigg Trinity College , Dublin Advocate Fiscal of Ceylon 7 Richard Ottley 1782 -- 1845 8 May 1820 -- 1833 1 November 1827 -- 1833 Barnes Chief Justice of Grenada 8 Henry Matthews England 1789 -- 1828 10 October 1827 -- 1828 -- -- Barnes King 's College , Cambridge Advocate Fiscal of Ceylon 9 Charles Marshall England 1788 - 1873 14 June 1828 -- 3 March 1836 18 February 1833 -- 3 March 1836 2,850 1,109 Barnes Jesus College , Cambridge 10 William Rough England 1772 - 1838 18 February 1833 - 1838 -- -- Trinity College , Cambridge 11 William Norris 1834 -- 1837 27 April 1836 -- 1837 Wilmot - Horton Advocate Fiscal of Ceylon 12 John Jeremie Guernsey 1795 - 1841 10 March 1836 -- 1840 -- -- Wilmot - Horton Public prosecutor of Mauritius 13 John Fredrick Stoddart Malta 1805 - 1839 2 June 1838 -- 19 August 1839 † -- 443 -- University of Glasgow 14 Anthony Oliphant Scotland 1793 -- 1859 22 October 1838 - 1854 22 October 1838 - 1854 Attorney General of the Cape Colony 15 J.G. Hildebrand 23 November 1839 -- -- -- 16 William Ogle Carr England 1802 - 1856 19 December 1939 - 1856 17 April 1854 -- 1856 Stewart - Mackenzie King 's Advocate of Ceylon 17 John Stark 10 October 1840 -- -- -- Queen 's Advocate of Ceylon 18 Christopher Temple 28 January 1854 -- -- -- Judge of the District Court of Colombo 19 Paul Ivy Sterling 20 November 1855 -- -- -- Attorney General of Hong Kong 20 William Carpenter Rowe - 1860 1857 -- 1859 -- 21 Edward Shepherd Creasy England 1812 - 1878 -- 27 March 1860 - 1875 University of Cambridge 22 Richard Morgan -- -- -- Queen 's Advocate of Ceylon 23 George Lawson -- -- -- 24 Henry Byerley Thompson - 1867 1 January 1863 -- -- -- Queen 's Advocate of Ceylon 25 Charles Henry Stewart 11 January 1867 -- -- -- 26 Richard Cayley England 26 July 1874 -- 1882 1 October 1879 -- 1882 -- Queen 's Advocate of Ceylon 27 Lovell Burchett Clarence 5 July 1879 -- -- -- 28 Harry Dias Bandaranaike 5 July 1879 -- -- -- 29 William Hackett 3 February 1877 - 1877 3 February 1877 - 1877 -- 30 John Budd Phear England 18 October 1877 -- -- -- 31 Archibald Campbell Lawrie 1837 - 1914 6 July 1892 -- 1901 -- -- Judge of the District Court of Kandy 32 Jacobus de Wet South Africa 31 May 1882 -- 29 May 1883 31 May 1882 -- 29 May 1883 363 363 33 Bruce Burnside Bahamas 21 May 1883 -- 1893 21 May 1883 -- 1893 -- Queen 's Advocate of Ceylon 34 John Winfield Bonser 13 November 1893 -- 1902 13 November 1893 -- 1902 Chief Justice of the Straits Settlements 35 George Henry Withers -- -- -- 36 Dodwell Browne -- -- -- 37 Fredrick Charles Moncrieff 4 April 1900 -- -- -- 38 Henry Lorentz Wendt 15 October 1901 -- 28 November 1910 -- -- 39 John Page Middleton 28 January 1902 -- 1912 -- -- Puisine Judge of the Supreme Court of Cyprus 40 Charles Layard Western 26 April 1902 -- 18 June 1906 26 April 1902 -- 18 June 1906 1,514 1,514 41 Joseph Richard Grenier 28 November 1910 -- 1912 -- -- 42 Alexander Wood Renton 3 September 1905 -- 1918 22 August 1914 - 1918 -- 43 Alfred George Lascelles England 1 May 1911 -- 1914 1 May 1911 -- 1914 -- 44 Joseph Turner Hutchinson England 23 October 1906 -- 1 May 1911 23 October 1906 -- 1 May 1911 1,651 1,651 45 James Cecil Walter Pereira 4 July 1912 -- -- -- 46 James Arthur Van Langenburg -- -- -- 47 Thomas De Sampayo 1915 -- -- -- University of Cambridge Commissioner of Assize 48 George Francis Macdaniel Ennis 5 August 1912 -- -- -- 49 Walter Shaw 10 December 1914 -- -- -- Chief Justice of British Honduras 50 Gualterus Stewart Schneider -- -- -- 51 Anton Bertram 1869 - 1937 26 July 1918 -- 1925 26 July 1918 -- 1925 University of Cambridge Attorney General of Ceylon 52 Hermann Albert Loos -- -- -- 53 William Thomas Porter 12 January 1922 -- -- -- 54 Thomas Garvin 1881 - 1940 21 June 1921 -- 1935 -- -- Manning Colombo Law College Solicitor General of Ceylon 55 John Adrian St. Valentine Jayewardene 1877 -- 1927 1923 -- -- -- Manning Judge of the District Court of Colombo 56 Llwellyn Chisholm Dalton 14 March 1925 -- -- -- 57 Lewis Matthew Maartensz 18 February 1935 -- -- -- 58 Maas Thajoon Akbar 1880 - 1944 -- -- -- University of Cambridge Solicitor General of Ceylon 59 Charles Ernest St. John Branch 1865 - 1939 3 July 1925 -- 25 May 1926 3 July 1925 -- 25 May 1926 326 326 Durham University 60 Stanley Fisher 1867 - 11 December 1926 -- 1930 11 December 1926 -- 1930 Chief Justice of Trinidad and Tobago 61 Robert William Lyall Grant 17 February 1926 -- -- -- 62 Allen Drieberg 19 September 1927 -- -- -- 63 Eugene Wilfred Jayewardene Western 1874 - 1932 May 1928 -- -- -- Fletcher Member of the Colombo Municipal Council 64 Philip James Macdonell 1873 - 1940 3 October 1930 - 1936 3 October 1930 - 1936 Chief Justice of Trinidad and Tobago 65 Lucian Macull Dominic de Silva -- -- -- 66 William Edward Barber -- -- -- 67 Francis Harold Bertram 19 February 1935 -- -- -- 68 Kenneth Poyser -- -- -- 69 Francis Soertsz - 1951 12 November 1936 -- -- -- 70 Sidney Abrahams England 1885 - 1957 3 July 1936 -- 1939 3 July 1936 -- 1939 University of Cambridge 71 Francis Moseley 2 August 1936 -- -- -- 72 Vincent Fernando 11 November 1937 -- -- -- 73 Hector Hearne 21 May 1937 -- -- -- 74 Arthur Eric Keuneman 4 March 1938 -- -- -- University of Cambridge 75 Oswald Leslie De Kretser II Western 1882 - 1959 8 April 1938 -- 1945 -- -- Caldecott University College , Colombo Commissioner of Assize 76 Arthur Wijewardena 1887 - 1964 22 August 1938 -- 1950 15 January 1949 -- 1950 University of Cambridge Solicitor General of Ceylon 77 J.H.B. Nihill 18 November 1938 -- -- -- 78 John Curtois Howard 1 December 1939 -- 1945 1 December 1939 -- 1945 Attorney General of Cyprus 79 G.H.F. Cannon -- -- -- 80 Edward Jayetileke 14 May 1942 -- 11 October 1951 1950 -- 11 October 1951 3,437 Caldecott Solicitor General of Ceylon 81 Alan Rose 11 January 1945 -- 1956 11 October 1951 -- 1956 82 A.R.H. Canakeratne 1886 - 1960 28 March 1945 -- -- -- Ceylon Law College University of London Commissioner of Assize 83 Manikku Wadumestri Hendrick de Silva -- -- -- Attorney General of Ceylon 84 Felix Reginald Dias Bandaranaike I 1861 - 1947 3 September 1946 -- -- -- University of Cambridge Judge of the District Court of Colombo 85 Chellappah Nagalingam 1893 - 1958 22 July 1947 -- 1958 -- -- Ceylon Law College Attorney General of Ceylon 86 Ralph Windham 23 July 1947 -- -- -- 87 Hema Henry Basnayake 1902 - 23 October 1947 -- 3 August 1964 1 January 1956 -- 3 August 1964 6,129 3,137 Senanayake Attorney General of Ceylon 88 Edward Frederick Noel Gratiaen 22 March 1948 -- -- -- Senanayake Attorney General of Ceylon 89 E.H.T. Gunasekera 1 March 1949 -- 25 January 1962 -- 4,713 -- Senanayake 90 M.F.S. Pulle 28 September 1949 -- -- -- Senanayake 91 V.L. St. C. Swan 1 May 1950 -- -- -- Senanayake 92 H.A. de Silva 30 November 1950 -- -- -- Senanayake 93 Kairshasp Choksy Western -- -- -- 94 Henry Weerasooriya 10 July 1953 -- 24 February 1964 -- 3,881 -- Senanayake 95 K.D. de Silva 10 July 1953 -- -- -- Senanayake 96 Miliani Sansoni 10 May 1954 -- 17 November 1966 3 August 1964 -- 17 November 1966 4,574 836 Kotelawala 97 Hugh Fernando 1910 - 1976 16 May 1955 -- 17 November 1973 20 November 1966 - 17 November 1973 6,760 2,554 Kotelawala 98 Thusew Samuel Fernando 2 May 1956 -- -- -- Bandaranaike Colombo Law College Solicitor General of Sri Lanka 99 Nadaraja Sinnetamby 3 May 1956 -- -- -- Bandaranaike 100 Lionel de Silva 5 November 1959 -- -- -- Dahanayake 101 Leonard de Silva 16 August 1966 -- -- -- Senanayake 102 Henry Thambiah Northern 1906 -- 1997 1 December 1961 -- 27 May 1968 -- 2,369 -- Bandaranaike University College , Colombo Ceylon Law College University of London Commissioner of Assize 103 Kingsley Herath 25 January 1962 -- -- -- Bandaranaike 104 Asoka Abeyesundere Southern 1906 - 1988 20 April 1962 -- 3 May 1968 -- 2,205 -- Bandaranaike Ceylon Law College 105 Ponnuduraiswamy Sriskandarajah 1905 - 1968 22 June 1962 -- 8 February 1967 -- 1,692 -- Bandaranaike Ceylon University College Ceylon Law College Commissioner of Assize 106 Gardiye Punchihewage Amaraseela Silva 22 June 1962 -- 1974 1973 -- 1974 -- Bandaranaike 107 Lionel Sirimanne -- -- -- Bandaranaike 108 Vaithalingam Manicavasagar Western 1906 - 1993 17 December 1964 -- 14 December 1967 -- 1,092 -- Bandaranaike Colombo Law College Commissioner of Assize 109 Anthony Alles Central 1911 -- 2003 1964 -- 1974 -- -- Bandaranaike University of Ceylon Ceylon Law College University of London Solicitor General of Sri Lanka 110 Veeravagu Sivasubramaniam 1908 - 1985 1966 -- 1970 -- -- Senanayake University of London Commissioner of Assize 111 George Samerawickrame 18 November 1966 -- 9 April 1981 -- 5,256 -- Senanayake 112 Victor Tennekoon Central 1914 - 8 February 1967 -- 1 July 1970 1 January 1974 - 8 September 1977 1 January 1974 - 8 September 1977 1,239 + 1,346 1,346 Senanayake University of London Ceylon Law College Judge of the Court of Appeal 113 Christopher Weeramantry Western 1926 - 1967 -- 30 April 1972 -- -- Senanayake University of Ceylon University of London Commissioner of Assize 114 Oswald Leslie De Kretser III Western 1910 - 1968 -- 21 January 1972 -- -- Senanayake University College , Colombo University of London Judge of the District Court of Panadura 115 Samarapilimudalige Wijayatilake Central 1912 - 1990 1968 -- 16 May 1975 -- -- Senanayake University College , Colombo Ceylon Law College Commissioner of Assize 116 Pandita Gunawadena 1968 -- 10 February 1971 -- -- Senanayake 117 Vincent Thamotheram 1915 - 23 March 1970 -- 1980 -- -- Senanayake Ceylon University College Ceylon Law College Commissioner of Assize 118 Hector Deheragoda 22 February 1972 -- 27 December 1976 -- 1,770 -- Bandaranaike 119 Cuda Banda Walgampaya 22 February 1972 -- 3 January 1976 -- 1,411 -- Bandaranaike 120 Jaya Pathirana - 2000 22 February 1972 -- 1978 -- -- Bandaranaike 121 Dharmadasa Wimalaratne 10 April 1972 - ? 20 October 1980 -- 15 September 1985 -- 1,791 -- Bandaranaike 122 Tellipalai Rajaratnam 1920 - 1994 10 April 1972 -- 1978 -- -- Bandaranaike Commissioner of Assize 123 Donald Sirimanne 1973 -- -- -- Gopallawa 124 Cyril Udalagama 1 January 1974 -- -- -- Gopallawa 125 T.A. de S. Wijesundera 1 January 1974 -- -- -- Gopallawa 126 Malcolm Perera 1 January 1974 -- -- -- Gopallawa 127 Izadean Mohamed Ismail 1 January 1974 -- 18 December 1981 -- 2,908 -- Gopallawa 128 Joe Weereratne 1 January 1974 -- -- -- Gopallawa 129 Arulampalam Vythialingam 1 January 1974 -- -- -- Gopallawa 130 Noel Tittawela 1 January 1974 -- -- -- Gopallawa 131 Suppiah Sharvananda Northern 1923 - 2007 1 January 1974 -- 1988 29 October 1984 -- 1988 Gopallawa Colombo Law College University of London 132 Somaratne Walpita 1 January 1974 -- -- -- Gopallawa 133 Wilmot Gunasekera 2 May 1974 -- -- -- Gopallawa 134 Barnes Ratwatte II 1918 - 2004 2 December 1974 -- ? 11 April 1981 -- 22 November 1983 -- 955 -- Gopallawa Jayewardene Ceylon Law College President of the Court of Appeal of Sri Lanka 135 Raja Wanasundera Sabaragamuwa 1924 - 2012 5 June 1975 -- ? -- -- Gopallawa University College , Colombo Ceylon Law College University of London Stanford University Solicitor General of Sri Lanka 136 Percy Colin - Thomé Southern 1922 - 12 January 1976 -- 1977 29 December 1981 -- ? -- -- Gopallawa Jayewardene University of Ceylon University of Cambridge Judge of the High Court of Colombo 137 Neville Samarakoon 1919 - 1990 -- 21 October 1984 1977 -- 21 October 1984 Jayewardene University College , Colombo Colombo Law College 138 Victor Perera 24 July 1981 -- 18 August 1983 -- 755 -- Jayewardene 139 Anton Soza Western 1919 - 2013 26 January 1982 -- 3 February 1984 -- 738 -- Jayewardene University of London Judge of the Court of Appeal 140 Parinda Ranasinghe 15 November 1982 -- 1991 1988 -- 1991 Jayewardene Colombo Law College 141 Abdul Cader 8 December 1982 -- 28 May 1985 -- 902 -- Jayewardene 142 Heevi Rodrigo 8 December 1982 -- 4 December 1984 -- 727 -- Jayewardene 143 E.A.D. Atukorale 27 November 1984 -- -- -- Jayewardene 144 Herbert Thambiah Northern 1926 -- 1992 27 November 1984 -- 1992 1991 -- 1991 Jayewardene University of Ceylon Ceylon Law College Judge of the Court of Appeal 145 Lucian Hector de Alwis 27 November 1984 -- -- -- Jayewardene 146 O.S.M. Seneviratne 10 February 1986 -- -- -- Jayewardene 147 H.A.G. de Silva 10 February 1986 -- -- -- Jayewardene Mark Fernando 1941 - 2009 4 March 1988 -- 2005 -- -- Jayewardene University of Ceylon , Peradeniya A.R.B. Amerasinghe 1937 - 2013 -- -- -- A.R.N. Fernando -- -- -- Ameer Ismail 1990 -- 1994 -- -- Premadasa University of Ceylon Ceylon Law College President of the Court of Appeal of Sri Lanka Andrew Ranjan Perera -- -- -- Pathmanathan Ramanathan 1932 - 2006 -- -- -- University of Wales President of the Court of Appeal of Sri Lanka K.M.M.B. Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28 January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,576 708 Kumaratunga University of Peradeniya University of Colombo University of Illinois at Urbana -- Champaign President of the Court of Appeal of Sri Lanka Nihal Jayasinghe 19 February 2003 -- 29 February 2008 -- 1,836 -- Kumaratunga University of Colombo Judge of the Court of Appeal Nimal Dissanayake -- 23 April 2008 -- -- Kumaratunga Shiranee Tilakawardane 19 February 2003 - 5 April 2014 -- 1,901 -- Kumaratunga Sri Lanka Law College President of the Court of Appeal of Sri Lanka Nimal Gamini Amaratunga January 2005 -- March 2014 -- -- Kumaratunga University of Colombo International Maritime Law Institute Judge of the Court of Appeal Saleem Marsoof February 2005 -- 2014 -- -- Kumaratunga University of Colombo , University of San Diego School of Law President of the Court of Appeal of Sri Lanka Andrew Somawansa North Western 28 May 2006 -- October 2008 -- -- Rajapaksa University of Ceylon University of Nevada President of the Court of Appeal of Sri Lanka Jagath Balapatabendi Southern -- -- -- Rajapaksa Ceylon Law College Judge of the Court of Appeal Kanagasabapathy Sripavan † 1952 -- 27 March 2008 -- present 30 January 2015 -- 28 February 2017 7003368200000000000 ♠ 3,682 7003118200000000000 ♠ 1,182 Rajapaksa Sri Lanka Law College Judge of the Court of Appeal P.A. Ratnayake 2 May 2008 -- 15 February 2013 -- 7003175100000000000 ♠ 1,751 -- Rajapaksa University of Brussels Chandra Ekanayake November 2008 -- 2016 -- -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka S.I. Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 -- present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,119 -- Rajapaksa Sri Lanka Law College University of Leicester Attorney General of Sri Lanka Mohan Peiris ♠ 15 January 2013 -- 28 January 2015 15 January 2013 -- 28 January 2015 743 743 Rajapaksa Sri Lanka Law College Attorney General of Sri Lanka Rohini Marasinghe 26 April 2013 -- 2015 -- -- Rajapaksa University of Hartford Judge of the Court of Appeal Buwaneka Aluwihare † 4 December 2013 -- present -- 7003160400000000000 ♠ 1,604 -- Rajapaksa University of London Additional Solicitor General of Sri Lanka Sisira de Abrew † Western 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne † 30 January 2015 -- present -- 7003118200000000000 ♠ 1,182 -- Sirisena Sri Lanka Law College University of Exeter Judge of the Court of Appeal Kankanithanthri Chitrasiri † 3 December 2015 -- present -- 7002875000000000000 ♠ 875 -- Sirisena University of Sri Lanka Queen Mary University of London Judge of the Court of Appeal Nalin Perera † 3 March 2016 -- present -- 7002784000000000000 ♠ 784 -- Sirisena Judge of the Court of Appeal Prasanna Sujeewa Jayawardena † 16 June 2016 -- present -- 7002679000000000000 ♠ 679 -- Sirisena University of Colombo Judge of the Court of Appeal
  Fact 0.0: Edmund Carrington was from England, born 1769 and died 1849.
  Fact 0.1: Edmund Carrington's active service term was March 1801 to 2 April 1806.
  Fact 0.2: Edmund Carrington's province was North.
  Fact 0.3: Edmund Carrington's prior position was counsel to the East India Company.
  Fact 0.4: Edmund Henry Lushington was from England, born 1766 and died 1839.
  Fact 0.5: Edmund Henry Lushington's active service term began 5 September 1801 and ended in 1809.
  Fact 0.6: Edmund Henry Lushington's term as chief was 15 April 1807 to 1809.
  Fact 0.7: Edmund Henry Lushington's province was North.
  Fact 0.8: Edmund Henry Lushington studied at the University of Cambridge.
  Fact 0.9: Edmund Henry Lushington's prior position was barrister.
  Fact 0.10: Alexander Johnston was from Scotland, born 1775 and died 1849.
  Fact 0.11: Alexander Johnston's active service term was 15 April 1807 to 14 February 1809.
  Fact 0.12: Alexander Johnston's term as chief was 6 November 1811 to 1819.
  Fact 0.13: Alexander Johnston was appointed by Maitland.
  Fact 0.14: Alexander Johnston studied at the University of Göttingen.
  Fact 0.15: Alexander Johnston's prior position was Advocate Fiscal of Ceylon.
  Fact 0.16: William Coke was from England, born 1776 and died 1818.
  Fact 0.17: William Coke's active service term was 6 March 1809 to 1 September 1818.
  Fact 0.18: William Coke's term as chief was 6 March 1809 to 1811.
  Fact 0.19: William Coke was appointed by Maitland.
  Fact 0.20: William Coke studied at the University of Oxford.
  Fact 0.21: William Coke's prior position was Advocate Fiscal of Ceylon.
  Fact 0.22: Henry Byrne was chief from 5 November 1818 to 1819.
  Fact 0.23: Henry Byrne was appointed by Brownrigg.
  Fact 0.24: Henry Byrne's prior position was barrister.
  Fact 0.25: Ambrose Hardinge Giffard was from Ireland, born 1771 and died 1827.
  Fact 0.26: Ambrose Hardinge Giffard's active service term was 8 April 1819 to 2 March 1827.
  Fact 0.27: Ambrose Hardinge Giffard was appointed by Brownrigg.
  Fact 0.28: Ambrose Hardinge Giffard studied at Trinity College, Dublin.
  Fact 0.29: Ambrose Hardinge Giffard's prior position was Advocate Fiscal of Ceylon.
  Fact 0.30: Richard Ottley lived from 1782 to 1845.
  Fact 0.31: Richard Ottley's active service term was 8 May 1820 to 1833.
  Fact 0.32: Richard Ottley's term as chief was 1 November 1827 to 1833.
  Fact 0.33: Richard Ottley was appointed by Barnes.
  Fact 0.34: Richard Ottley's prior position was Chief Justice of Grenada.
  Fact 0.35: Henry Matthews was from England, born 1789 and died 1828.
  Fact 0.36: Henry Matthews's active service term was 10 October 1827 to 1828.
  Fact 0.37: Henry Matthews was appointed by Barnes.
  Fact 0.38: Henry Matthews studied at King's College, Cambridge.
  Fact 0.39: Henry Matthews's prior position was Advocate Fiscal of Ceylon.
  Fact 0.40: Charles Marshall was from England, born 1788 and died 1873.
  Fact 0.41: Charles Marshall's active service term was 14 June 1828 to 3 March 1836.
  Fact 0.42: Charles Marshall's term as chief was 18 February 1833 to 3 March 1836.
  Fact 0.43: Charles Marshall was appointed by Barnes.
  Fact 0.44: Charles Marshall studied at Jesus College, Cambridge.
  Fact 0.45: William Rough was from England, born 1772 and died 1838.
  Fact 0.46: William Rough's term was 18 February 1833 to 1838.
  Fact 0.47: William Rough studied at Trinity College, Cambridge.
  Fact 0.48: William Norris's term was 27 April 1836 to 1837.
  Fact 0.49: William Norris was appointed by Wilmot-Horton.
  Fact 0.50: William Norris's prior position was Advocate Fiscal of Ceylon.
  Fact 0.51: John Jeremie was from Guernsey, born 1795 and died 1841.
  Fact 0.52: John Jeremie's term was 10 March 1836 to 1840.
  Fact 0.53: John Jeremie was appointed by Wilmot-Horton.
  Fact 0.54: John Jeremie's prior position was public prosecutor of Mauritius.
  Fact 0.55: John Fredrick Stoddart was from Malta, born 1805 and died 1839.
  Fact 0.56: John Fredrick Stoddart's active service term was 2 June 1838 to 19 August 1839.
  Fact 0.57: John Fredrick Stoddart's length of active service was 443 days.
  Fact 0.58: John Fredrick Stoddart studied at the University of Glasgow.
  Fact 0.59: Anthony Oliphant was from Scotland, born 1793 and died 1859.
  Fact 0.60: Anthony Oliphant's active service term was 22 October 1838 to 1854.
  Fact 0.61: Anthony Oliphant's term as chief was 22 October 1838 to 1854.
  Fact 0.62: Anthony Oliphant's prior position was Attorney General of the Cape Colony.
  Fact 0.63: J.G. Hildebrand's term began 23 November 1839.
  Fact 0.64: William Ogle Carr was from England, born 1802 and died 1856.
  Fact 0.65: William Ogle Carr's term was 17 April 1854 to 1856.
  Fact 0.66: William Ogle Carr was appointed by Stewart-Mackenzie.
  Fact 0.67: William Ogle Carr's prior position was King's Advocate of Ceylon.
  Fact 0.68: John Stark's term began 10 October 1840.
  Fact 0.69: John Stark's prior position was Queen's Advocate of Ceylon.
  Fact 0.70: Christopher Temple's term began 28 January 1854.
  Fact 0.71: Christopher Temple's prior position was judge of the District Court of Colombo.
  Fact 0.72: Paul Ivy Sterling's term began 20 November 1855.
  Fact 0.73: Paul Ivy Sterling's prior position was Attorney General of Hong Kong.
  Fact 0.74: William Carpenter Rowe died in 1860.
  Fact 0.75: William Carpenter Rowe's term was 1857 to 1859.
  Fact 0.76: Edward Shepherd Creasy was from England, born 1812 and died 1878.
  Fact 0.77: Edward Shepherd Creasy's term was 27 March 1860 to 1875.
  Fact 0.78: Edward Shepherd Creasy studied at the University of Cambridge.
  Fact 0.79: Richard Morgan's prior position was Queen's Advocate of Ceylon.
  Fact 0.80: George Lawson has no recorded dates in the table.
  Fact 0.81: Henry Byerley Thompson died in 1867.
  Fact 0.82: Henry Byerley Thompson's term began 1 January 1863.
  Fact 0.83: Henry Byerley Thompson's prior position was Queen's Advocate of Ceylon.
  Fact 0.84: Charles Henry Stewart's term began 11 January 1867.
  Fact 0.85: Richard Cayley was from England.
  Fact 0.86: Richard Cayley's term was 26 July 1874 to 1882.
  Fact 0.87: Richard Cayley's term as chief was 1 October 1879 to 1882.
  Fact 0.88: Richard Cayley's prior position was Queen's Advocate of Ceylon.
  Fact 0.89: Lovell Burchett Clarence's term began 5 July 1879.
  Fact 0.90: Harry Dias Bandaranaike's term began 5 July 1879.
  Fact 0.91: William Hackett's term was 3 February 1877 to 1877.
  Fact 0.92: John Budd Phear was from England and his term began 18 October 1877.
  Fact 0.93: Archibald Campbell Lawrie lived from 1837 to 1914.
  Fact 0.94: Archibald Campbell Lawrie's term was 6 July 1892 to 1901.
  Fact 0.95: Archibald Campbell Lawrie's prior position was judge of the District Court of Kandy.
  Fact 0.96: Jacobus de Wet was from South Africa.
  Fact 0.97: Jacobus de Wet's term was 31 May 1882 to 29 May 1883, a length of 363 days.
  Fact 0.98: Bruce Burnside was from the Bahamas.
  Fact 0.99: Bruce Burnside's term was 21 May 1883 to 1893.
  Fact 0.100: Bruce Burnside's prior position was Queen's Advocate of Ceylon.
  Fact 0.101: John Winfield Bonser's term was 13 November 1893 to 1902.
  Fact 0.102: John Winfield Bonser's prior position was Chief Justice of the Straits Settlements.
  Fact 0.103: George Henry Withers has no recorded dates in the table.
  Fact 0.104: Dodwell Browne has no recorded dates in the table.
  Fact 0.105: Fredrick Charles Moncrieff's term began 4 April 1900.
  Fact 0.106: Henry Lorentz Wendt's term was 15 October 1901 to 28 November 1910.
  Fact 0.107: John Page Middleton's term was 28 January 1902 to 1912.
  Fact 0.108: John Page Middleton's prior position was judge of the Supreme Court of Cyprus.
  Fact 0.109: Charles Layard Western's term was 26 April 1902 to 18 June 1906, a length of 1,514 days.
  Fact 0.110: Joseph Richard Grenier's term was 28 November 1910 to 1912.
  Fact 0.111: Alexander Wood Renton's term was 3 September 1905 to 1918.
  Fact 0.112: Alexander Wood Renton's term as chief was 22 August 1914 to 1918.
  Fact 0.113: Alfred George Lascelles was from England and his term was 1 May 1911 to 1914.
  Fact 0.114: Joseph Turner Hutchinson was from England and his term was 23 October 1906 to 1 May 1911, a length of 1,651 days.
  Fact 0.115: James Cecil Walter Pereira's term began 4 July 1912.
  Fact 0.116: James Arthur Van Langenburg has no recorded dates in the table.
  Fact 0.117: Thomas De Sampayo's term began in 1915.
  Fact 0.118: Thomas De Sampayo studied at the University of Cambridge.
  Fact 0.119: Thomas De Sampayo's prior position was Commissioner of Assize.
  Fact 0.120: George Francis Macdaniel Ennis's term began 5 August 1912.
  Fact 0.121: Walter Shaw's term began 10 December 1914.
  Fact 0.122: Walter Shaw's prior position was Chief Justice of British Honduras.
  Fact 0.123: Gualterus Stewart Schneider has no recorded dates in the table.
  Fact 0.124: Anton Bertram lived from 1869 to 1937.
  Fact 0.125: Anton Bertram's term was 26 July 1918 to 1925.
  Fact 0.126: Anton Bertram studied at the University of Cambridge.
  Fact 0.127: Anton Bertram's prior position was Attorney General of Ceylon.
  Fact 0.128: Hermann Albert Loos has no recorded dates in the table.
  Fact 0.129: William Thomas Porter's term began 12 January 1922.
  Fact 0.130: Thomas Garvin lived from 1881 to 1940.
  Fact 0.131: Thomas Garvin's term was 21 June 1921 to 1935.
  Fact 0.132: Thomas Garvin studied at Colombo Law College.
  Fact 0.133: Thomas Garvin's prior position was Solicitor General of Ceylon.
  Fact 0.134: John Adrian St. Valentine Jayewardene lived from 1877 to 1927.
  Fact 0.135: John Adrian St. Valentine Jayewardene's term began in 1923.
  Fact 0.136: John Adrian St. Valentine Jayewardene was appointed by Manning.
  Fact 0.137: John Adrian St. Valentine Jayewardene's prior position was judge of the District Court of Colombo.
  Fact 0.138: Llwellyn Chisholm Dalton's term began 14 March 1925.
  Fact 0.139: Lewis Matthew Maartensz's term began 18 February 1935.
  Fact 0.140: Maas Thajoon Akbar lived from 1880 to 1944.
  Fact 0.141: Maas Thajoon Akbar studied at the University of Cambridge.
  Fact 0.142: Maas Thajoon Akbar's prior position was Solicitor General of Ceylon.
  Fact 0.143: Charles Ernest St. John Branch lived from 1865 to 1939.
  Fact 0.144: Charles Ernest St. John Branch's term was 3 July 1925 to 25 May 1926, a length of 326 days.
  Fact 0.145: Charles Ernest St. John Branch studied at Durham University.
  Fact 0.146: Stanley Fisher was born in 1867.
  Fact 0.147: Stanley Fisher's term was 11 December 1926 to 1930.
  Fact 0.148: Stanley Fisher's prior position was Chief Justice of Trinidad and Tobago.
  Fact 0.149: Robert William Lyall Grant's term began 17 February 1926.
  Fact 0.150: Allen Drieberg's term began 19 September 1927.
  Fact 0.151: Eugene Wilfred Jayewardene lived from 1874 to 1932.
  Fact 0.152: Eugene Wilfred Jayewardene's term began in May 1928.
  Fact 0.153: Eugene Wilfred Jayewardene's prior position was member of the Colombo Municipal Council.
  Fact 0.154: Philip James Macdonell lived from 1873 to 1940.
  Fact 0.155: Philip James Macdonell's term was 3 October 1930 to 1936.
  Fact 0.156: Philip James Macdonell's prior position was Chief Justice of Trinidad and Tobago.
  Fact 0.157: Lucian Macull Dominic de Silva has no recorded dates in the table.
  Fact 0.158: William Edward Barber has no recorded dates in the table.
  Fact 0.159: Francis Harold Bertram's term began 19 February 1935.
  Fact 0.160: Kenneth Poyser has no recorded dates in the table.
  Fact 0.161: Francis Soertsz's term began 12 November 1936.
  Fact 0.162: Sidney Abrahams was from England, born 1885 and died 1957.
  Fact 0.163: Sidney Abrahams's term was 3 July 1936 to 1939.
  Fact 0.164: Sidney Abrahams studied at the University of Cambridge.
  Fact 0.165: Francis Moseley's term began 2 August 1936.
  Fact 0.166: Vincent Fernando's term began 11 November 1937.
  Fact 0.167: Hector Hearne's term began 21 May 1937.
  Fact 0.168: Arthur Eric Keuneman's term began 4 March 1938.
  Fact 0.169: Arthur Eric Keuneman studied at the University of Cambridge.
  Fact 0.170: Oswald Leslie De Kretser II lived from 1882 to 1959.
  Fact 0.171: Oswald Leslie De Kretser II's term was 8 April 1938 to 1945.
  Fact 0.172: Oswald Leslie De Kretser II studied at Caldecott and University College, Colombo.
  Fact 0.173: Oswald Leslie De Kretser II's prior position was Commissioner of Assize.
  Fact 0.174: Arthur Wijewardena lived from 1887 to 1964.
  Fact 0.175: Arthur Wijewardena's term was 22 August 1938 to 1950.
  Fact 0.176: Arthur Wijewardena's term as chief was 15 January 1949 to 1950.
  Fact 0.177: Arthur Wijewardena studied at the University of Cambridge.
  Fact 0.178: Arthur Wijewardena's prior position was Solicitor General of Ceylon.
  Fact 0.179: J.H.B. Nihill's term began 18 November 1938.
  Fact 0.180: John Curtois Howard's term was 1 December 1939 to 1945.
  Fact 0.181: John Curtois Howard's prior position was Attorney General of Cyprus.
  Fact 0.182: G.H.F. Cannon has no recorded dates in the table.
  Fact 0.183: Edward Jayetileke's term was 14 May 1942 to 11 October 1951, a length of 3,437 days.
  Fact 0.184: Edward Jayetileke studied at Caldecott.
  Fact 0.185: Edward Jayetileke's prior position was Solicitor General of Ceylon.
  Fact 0.186: Alan Rose's term was 11 January 1945 to 1956.
  Fact 0.187: A.R.H. Canakeratne lived from 1886 to 1960.
  Fact 0.188: A.R.H. Canakeratne's term began 28 March 1945.
  Fact 0.189: A.R.H. Canakeratne studied at Ceylon Law College and the University of London.
  Fact 0.190: A.R.H. Canakeratne's prior position was Commissioner of Assize.
  Fact 0.191: Manikku Wadumestri Hendrick de Silva has no recorded dates in the table.
  Fact 0.192: Manikku Wadumestri Hendrick de Silva's prior position was Attorney General of Ceylon.
  Fact 0.193: Felix Reginald Dias Bandaranaike I lived from 1861 to 1947.
  Fact 0.194: Felix Reginald Dias Bandaranaike I's term began 3 September 1946.
  Fact 0.195: Felix Reginald Dias Bandaranaike I studied at the University of Cambridge.
  Fact 0.196: Felix Reginald Dias Bandaranaike I's prior position was judge of the District Court of Colombo.
  Fact 0.197: Chellappah Nagalingam lived from 1893 to 1958.
  Fact 0.198: Chellappah Nagalingam's term was 22 July 1947 to 1958.
  Fact 0.199: Chellappah Nagalingam studied at Ceylon Law College.
  Fact 0.200: Chellappah Nagalingam's prior position was Attorney General of Ceylon.
  Fact 0.201: Ralph Windham's term began 23 July 1947.
  Fact 0.202: Hema Henry Basnayake was born in 1902 and his term began 23 October 1947.
  Fact 0.203: Hema Henry Basnayake's term ended 3 August 1964, a length of 6,129 days.
  Fact 0.204: Hema Henry Basnayake's term as chief was 1 January 1956 to 3 August 1964, a length of 3,137 days.
  Fact 0.205: Hema Henry Basnayake's prior position was Attorney General of Ceylon.
  Fact 0.206: Edward Frederick Noel Gratiaen's term began 22 March 1948.
  Fact 0.207: E.H.T. Gunasekera's term was 1 March 1949 to 25 January 1962, a length of 4,713 days.
  Fact 0.208: M.F.S. Pulle's term began 28 September 1949.
  Fact 0.209: V.L. St. C. Swan's term began 1 May 1950.
  Fact 0.210: H.A. de Silva's term began 30 November 1950.
  Fact 0.211: Kairshasp Choksy has no recorded dates in the table.
  Fact 0.212: Henry Weerasooriya's term was 10 July 1953 to 24 February 1964, a length of 3,881 days.
  Fact 0.213: K.D. de Silva's term began 10 July 1953.
  Fact 0.214: Miliani Sansoni's term was 10 May 1954 to 17 November 1966, a length of 4,574 days.
  Fact 0.215: Miliani Sansoni's term as chief was 3 August 1964 to 17 November 1966, a length of 836 days.
  Fact 0.216: Hugh Fernando lived from 1910 to 1976.
  Fact 0.217: Hugh Fernando's term was 16 May 1955 to 17 November 1973, a length of 6,760 days.
  Fact 0.218: Hugh Fernando's term as chief was 20 November 1966 to 17 November 1973, a length of 2,554 days.
  Fact 0.219: Thusew Samuel Fernando's term began 2 May 1956.
  Fact 0.220: Thusew Samuel Fernando studied at Colombo Law College.
  Fact 0.221: Thusew Samuel Fernando's prior position was Solicitor General of Sri Lanka.
  Fact 0.222: Nadaraja Sinnetamby's term began 3 May 1956.
  Fact 0.223: Lionel de Silva's term began 5 November 1959.
  Fact 0.224: Leonard de Silva's term began 16 August 1966.
  Fact 0.225: Henry Thambiah was from the Northern province, born 1906 and died 1997.
  Fact 0.226: Henry Thambiah's term was 1 December 1961 to 27 May 1968, a length of 2,369 days.
  Fact 0.227: Henry Thambiah studied at University College, Colombo, Ceylon Law College and the University of London.
  Fact 0.228: Henry Thambiah's prior position was Commissioner of Assize.
  Fact 0.229: Kingsley Herath's term began 25 January 1962.
  Fact 0.230: Asoka Abeyesundere was from the Southern province, born 1906 and died 1988.
  Fact 0.231: Asoka Abeyesundere's term was 20 April 1962 to 3 May 1968, a length of 2,205 days.
  Fact 0.232: Asoka Abeyesundere studied at Ceylon Law College.
  Fact 0.233: Ponnuduraiswamy Sriskandarajah lived from 1905 to 1968.
  Fact 0.234: Ponnuduraiswamy Sriskandarajah's term was 22 June 1962 to 8 February 1967, a length of 1,692 days.
  Fact 0.235: Ponnuduraiswamy Sriskandarajah studied at Ceylon University College and Ceylon Law College.
  Fact 0.236: Ponnuduraiswamy Sriskandarajah's prior position was Commissioner of Assize.
  Fact 0.237: Gardiye Punchihewage Amaraseela Silva's term was 22 June 1962 to 1974.
  Fact 0.238: Gardiye Punchihewage Amaraseela Silva's term as chief was 1973 to 1974.
  Fact 0.239: Lionel Sirimanne has no recorded dates in the table.
  Fact 0.240: Vaithalingam Manicavasagar was from the Western province, born 1906 and died 1993.
  Fact 0.241: Vaithalingam Manicavasagar's term was 17 December 1964 to 14 December 1967, a length of 1,092 days.
  Fact 0.242: Vaithalingam Manicavasagar studied at Colombo Law College.
  Fact 0.243: Vaithalingam Manicavasagar's prior position was Commissioner of Assize.
  Fact 0.244: Anthony Alles was from the Central province, born 1911 and died 2003.
  Fact 0.245: Anthony Alles's term was 1964 to 1974.
  Fact 0.246: Anthony Alles studied at the University of Ceylon, Ceylon Law College and the University of London.
  Fact 0.247: Anthony Alles's prior position was Solicitor General of Sri Lanka.
  Fact 0.248: Veeravagu Sivasubramaniam lived from 1908 to 1985.
  Fact 0.249: Veeravagu Sivasubramaniam's term was 1966 to 1970.
  Fact 0.250: Veeravagu Sivasubramaniam studied at the University of London.
  Fact 0.251: Veeravagu Sivasubramaniam's prior position was Commissioner of Assize.
  Fact 0.252: George Samerawickrame's term was 18 November 1966 to 9 April 1981, a length of 5,256 days.
  Fact 0.253: Victor Tennekoon was from the Central province and was born in 1914.
  Fact 0.254: Victor Tennekoon's term as chief was 1 January 1974 to 8 September 1977.
  Fact 0.255: Victor Tennekoon studied at the University of London and Ceylon Law College.
  Fact 0.256: Victor Tennekoon's prior position was Judge of the Court of Appeal.
  Fact 0.257: Christopher Weeramantry was from the Western province, born 1926 and died 1967.
  Fact 0.258: Christopher Weeramantry's term began 30 April 1972.
  Fact 0.259: Christopher Weeramantry studied at the University of Ceylon and the University of London.
  Fact 0.260: Christopher Weeramantry's prior position was Commissioner of Assize.
  Fact 0.261: Oswald Leslie De Kretser III was from the Western province, born 1910 and died 1968.
  Fact 0.262: Oswald Leslie De Kretser III's term began 21 January 1972.
  Fact 0.263: Oswald Leslie De Kretser III studied at University College, Colombo and the University of London.
  Fact 0.264: Oswald Leslie De Kretser III's prior position was judge of the District Court of Panadura.
  Fact 0.265: Samarapilimudalige Wijayatilake was from the Central province, born 1912 and died 1990.
  Fact 0.266: Samarapilimudalige Wijayatilake's term was 1968 to 16 May 1975.
  Fact 0.267: Samarapilimudalige Wijayatilake studied at University College, Colombo and Ceylon Law College.
  Fact 0.268: Samarapilimudalige Wijayatilake's prior position was Commissioner of Assize.
  Fact 0.269: Pandita Gunawadena's term was 1968 to 10 February 1971.
  Fact 0.270: Vincent Thamotheram was born in 1915 and died on 23 March 1970.
  Fact 0.271: Vincent Thamotheram studied at Ceylon University College and Ceylon Law College.
  Fact 0.272: Vincent Thamotheram's prior position was Commissioner of Assize.
  Fact 0.273: Hector Deheragoda's term was 22 February 1972 to 27 December 1976, a length of 1,770 days.
  Fact 0.274: Cuda Banda Walgampaya's term was 22 February 1972 to 3 January 1976, a length of 1,411 days.
  Fact 0.275: Jaya Pathirana died in 2000.
  Fact 0.276: Jaya Pathirana's term was 22 February 1972 to 1978.
  Fact 0.277: Dharmadasa Wimalaratne's term began 10 April 1972.
  Fact 0.278: Dharmadasa Wimalaratne's term as chief was 20 October 1980 to 15 September 1985, a length of 1,791 days.
  Fact 0.279: Tellipalai Rajaratnam lived from 1920 to 1994.
  Fact 0.280: Tellipalai Rajaratnam's term was 10 April 1972 to 1978.
  Fact 0.281: Tellipalai Rajaratnam's prior position was Commissioner of Assize.
  Fact 0.282: Donald Sirimanne's term began in 1973.
  Fact 0.283: Cyril Udalagama's term began 1 January 1974.
  Fact 0.284: T.A. de S. Wijesundera's term began 1 January 1974.
  Fact 0.285: Malcolm Perera's term began 1 January 1974.
  Fact 0.286: Izadean Mohamed Ismail's term was 1 January 1974 to 18 December 1981, a length of 2,908 days.
  Fact 0.287: Joe Weereratne's term began 1 January 1974.
  Fact 0.288: Arulampalam Vythialingam's term began 1 January 1974.
  Fact 0.289: Noel Tittawela's term began 1 January 1974.
  Fact 0.290: Suppiah Sharvananda was from the Northern province, born 1923 and died 2007.
  Fact 0.291: Suppiah Sharvananda's term was 1 January 1974 to 1988.
  Fact 0.292: Suppiah Sharvananda's term as chief was 29 October 1984 to 1988.
  Fact 0.293: Suppiah Sharvananda studied at Colombo Law College and the University of London.
  Fact 0.294: Somaratne Walpita's term began 1 January 1974.
  Fact 0.295: Wilmot Gunasekera's term began 2 May 1974.
  Fact 0.296: Barnes Ratwatte II lived from 1918 to 2004.
  Fact 0.297: Barnes Ratwatte II's term began 2 December 1974.
  Fact 0.298: Barnes Ratwatte II's term as chief was 11 April 1981 to 22 November 1983, a length of 955 days.
  Fact 0.299: Barnes Ratwatte II studied at Ceylon Law College.
  Fact 0.300: Barnes Ratwatte II's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.301: Raja Wanasundera was from the Sabaragamuwa province, born 1924 and died 2012.
  Fact 0.302: Raja Wanasundera's term began 5 June 1975.
  Fact 0.303: Raja Wanasundera studied at University College, Colombo, Ceylon Law College, the University of London and Stanford University.
  Fact 0.304: Raja Wanasundera's prior position was Solicitor General of Sri Lanka.
  Fact 0.305: Percy Colin-Thomé was from the Southern province, born 1922.
  Fact 0.306: Percy Colin-Thomé's term began 12 January 1976 and ran to 1977.
  Fact 0.307: Percy Colin-Thomé's term as chief began 29 December 1981.
  Fact 0.308: Percy Colin-Thomé studied at the University of Ceylon and the University of Cambridge.
  Fact 0.309: Percy Colin-Thomé's prior position was judge of the High Court of Colombo.
  Fact 0.310: Neville Samarakoon lived from 1919 to 1990.
  Fact 0.311: Neville Samarakoon's term as chief was 1977 to 21 October 1984.
  Fact 0.312: Neville Samarakoon studied at University College, Colombo and Colombo Law College.
  Fact 0.313: Victor Perera's term was 24 July 1981 to 18 August 1983, a length of 755 days.
  Fact 0.314: Anton Soza was from the Western province, born 1919 and died 2013.
  Fact 0.315: Anton Soza's term was 26 January 1982 to 3 February 1984, a length of 738 days.
  Fact 0.316: Anton Soza studied at the University of London.
  Fact 0.317: Anton Soza's prior position was Judge of the Court of Appeal.
  Fact 0.318: Parinda Ranasinghe's term was 15 November 1982 to 1991.
  Fact 0.319: Parinda Ranasinghe's term as chief was 1988 to 1991.
  Fact 0.320: Parinda Ranasinghe studied at Colombo Law College.
  Fact 0.321: Abdul Cader's term was 8 December 1982 to 28 May 1985, a length of 902 days.
  Fact 0.322: Heevi Rodrigo's term was 8 December 1982 to 4 December 1984, a length of 727 days.
  Fact 0.323: E.A.D. Atukorale's term began 27 November 1984.
  Fact 0.324: Herbert Thambiah was from the Northern province, born 1926 and died 1992.
  Fact 0.325: Herbert Thambiah's term began 27 November 1984 and ended in 1992.
  Fact 0.326: Herbert Thambiah's term as chief was in 1991.
  Fact 0.327: Herbert Thambiah studied at the University of Ceylon and Ceylon Law College.
  Fact 0.328: Herbert Thambiah's prior position was Judge of the Court of Appeal.
  Fact 0.329: Lucian Hector de Alwis's term began 27 November 1984.
  Fact 0.330: O.S.M. Seneviratne's term began 10 February 1986.
  Fact 0.331: H.A.G. de Silva's term began 10 February 1986.
  Fact 0.332: Mark Fernando lived from 1941 to 2009.
  Fact 0.333: Mark Fernando's term began 4 March 1988 and ended in 2005.
  Fact 0.334: Mark Fernando studied at the University of Ceylon, Peradeniya.
  Fact 0.335: A.R.B. Amerasinghe lived from 1937 to 2013.
  Fact 0.336: A.R.N. Fernando has no recorded dates in the table.
  Fact 0.337: Ameer Ismail's term was 1990 to 1994.
  Fact 0.338: Ameer Ismail studied at the University of Ceylon and Ceylon Law College.
  Fact 0.339: Ameer Ismail's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.340: Andrew Ranjan Perera has no recorded dates in the table.
  Fact 0.341: Pathmanathan Ramanathan lived from 1932 to 2006.
  Fact 0.342: Pathmanathan Ramanathan studied at the University of Wales.
  Fact 0.343: Pathmanathan Ramanathan's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.344: K.M.M.B. Kulatunge has no recorded dates in the table.
  Fact 0.345: G.P.S. de Silva's term was 1991 to 1999.
  Fact 0.346: Sarath N. Silva's term began 16 September 1999 and ended 7 June 2009, a length of 3,552 days.
  Fact 0.347: Sarath N. Silva was appointed by Kumaratunga.
  Fact 0.348: Sarath N. Silva studied at Sri Lanka Law College and the University of Brussels.
  Fact 0.349: Sarath N. Silva's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.350: Shirani Bandaranayake was from the North Western province and was born in 1958.
  Fact 0.351: Shirani Bandaranayake was appointed by Kumaratunga.
  Fact 0.352: Shirani Bandaranayake studied at the University of Colombo and the University of London.
  Fact 0.353: Shirani Bandaranayake's prior position was lawyer.
  Fact 0.354: C.V. Vigneswaran was from the Western province, born 1939 and died in March 2001.
  Fact 0.355: C.V. Vigneswaran's term began in October 2004.
  Fact 0.356: C.V. Vigneswaran was appointed by Kumaratunga.
  Fact 0.357: C.V. Vigneswaran studied at the University of Ceylon and Ceylon Law College.
  Fact 0.358: C.V. Vigneswaran's prior position was Judge of the Court of Appeal.
  Fact 0.359: Asoka de Silva lived from 1946 to 1 August 2001.
  Fact 0.360: Asoka de Silva's term began 17 May 2011.
  Fact 0.361: Asoka de Silva's active service length was 3,576 days.
  Fact 0.362: Asoka de Silva's term as chief was 8 June 2009 to 17 May 2011, a length of 708 days.
  Fact 0.363: Asoka de Silva was appointed by Kumaratunga.
  Fact 0.364: Asoka de Silva studied at the University of Peradeniya, the University of Colombo and the University of Illinois at Urbana-Champaign.
  Fact 0.365: Asoka de Silva's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.366: Nihal Jayasinghe's term was 19 February 2003 to 29 February 2008, a length of 1,836 days.
  Fact 0.367: Nihal Jayasinghe was appointed by Kumaratunga.
  Fact 0.368: Nihal Jayasinghe studied at the University of Colombo.
  Fact 0.369: Nihal Jayasinghe's prior position was Judge of the Court of Appeal.
  Fact 0.370: Nimal Dissanayake's term began 23 April 2008.
  Fact 0.371: Nimal Dissanayake was appointed by Kumaratunga.
  Fact 0.372: Shiranee Tilakawardane's term was 19 February 2003 to 5 April 2014, a length of 1,901 days.
  Fact 0.373: Shiranee Tilakawardane was appointed by Kumaratunga.
  Fact 0.374: Shiranee Tilakawardane studied at Sri Lanka Law College.
  Fact 0.375: Shiranee Tilakawardane's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.376: Nimal Gamini Amaratunga's term began in January 2005 and ended in March 2014.
  Fact 0.377: Nimal Gamini Amaratunga was appointed by Kumaratunga.
  Fact 0.378: Nimal Gamini Amaratunga studied at the University of Colombo and the International Maritime Law Institute.
  Fact 0.379: Nimal Gamini Amaratunga's prior position was Judge of the Court of Appeal.
  Fact 0.380: Saleem Marsoof's term began in February 2005 and ended in 2014.
  Fact 0.381: Saleem Marsoof was appointed by Kumaratunga.
  Fact 0.382: Saleem Marsoof studied at the University of Colombo and the University of San Diego School of Law.
  Fact 0.383: Saleem Marsoof's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.384: Andrew Somawansa was from the North Western province.
  Fact 0.385: Andrew Somawansa's term began 28 May 2006 and ended in October 2008.
  Fact 0.386: Andrew Somawansa was appointed by Rajapaksa.
  Fact 0.387: Andrew Somawansa studied at the University of Ceylon and the University of Nevada.
  Fact 0.388: Andrew Somawansa's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.389: Jagath Balapatabendi was from the Southern province.
  Fact 0.390: Jagath Balapatabendi was appointed by Rajapaksa.
  Fact 0.391: Jagath Balapatabendi studied at Ceylon Law College.
  Fact 0.392: Jagath Balapatabendi's prior position was Judge of the Court of Appeal.
  Fact 0.393: Kanagasabapathy Sripavan was born in 1952.
  Fact 0.394: Kanagasabapathy Sripavan's term began 27 March 2008.
  Fact 0.395: Kanagasabapathy Sripavan's term as chief was 30 January 2015 to 28 February 2017.
  Fact 0.396: Kanagasabapathy Sripavan was appointed by Rajapaksa.
  Fact 0.397: Kanagasabapathy Sripavan studied at Sri Lanka Law College.
  Fact 0.398: Kanagasabapathy Sripavan's prior position was Judge of the Court of Appeal.
  Fact 0.399: P.A. Ratnayake's term was 2 May 2008 to 15 February 2013, a length of 1,751 days.
  Fact 0.400: P.A. Ratnayake was appointed by Rajapaksa.
  Fact 0.401: P.A. Ratnayake studied at the University of Brussels.
  Fact 0.402: Chandra Ekanayake's term was November 2008 to 2016.
  Fact 0.403: Chandra Ekanayake was appointed by Rajapaksa.
  Fact 0.404: Chandra Ekanayake studied at Sri Lanka Law College.
  Fact 0.405: Chandra Ekanayake's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.406: S.I. Imam was from the Central province.
  Fact 0.407: S.I. Imam's term was 9 January 2009 to 20 February 2013, a length of 1,503 days.
  Fact 0.408: S.I. Imam was appointed by Rajapaksa.
  Fact 0.409: S.I. Imam's prior position was Judge of the Court of Appeal.
  Fact 0.410: R.K.S. Suresh Chandra's term began 17 June 2010.
  Fact 0.411: R.K.S. Suresh Chandra was appointed by Rajapaksa.
  Fact 0.412: Sathya Hettige's term was 10 July 2011 to 18 April 2014, a length of 1,013 days.
  Fact 0.413: Sathya Hettige was appointed by Rajapaksa.
  Fact 0.414: Sathya Hettige studied at the University of Colombo.
  Fact 0.415: Sathya Hettige's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.416: Priyasath Dep's term began 10 July 2011 and is ongoing.
  Fact 0.417: Priyasath Dep's term as chief began 2 March 2017.
  Fact 0.418: Priyasath Dep was appointed by Rajapaksa.
  Fact 0.419: Priyasath Dep studied at the University of Colombo and the International Institute of Social Studies.
  Fact 0.420: Priyasath Dep's prior position was Solicitor General of Sri Lanka.
  Fact 0.421: Eva Wansundera was from the North Western province.
  Fact 0.422: Eva Wansundera's term began 7 July 2012 and is ongoing.
  Fact 0.423: Eva Wansundera was appointed by Rajapaksa.
  Fact 0.424: Eva Wansundera studied at Sri Lanka Law College and the University of Leicester.
  Fact 0.425: Eva Wansundera's prior position was Attorney General of Sri Lanka.
  Fact 0.426: Mohan Peiris's term was 15 January 2013 to 28 January 2015, a length of 743 days.
  Fact 0.427: Mohan Peiris was appointed by Rajapaksa.
  Fact 0.428: Mohan Peiris studied at Sri Lanka Law College.
  Fact 0.429: Mohan Peiris's prior position was Attorney General of Sri Lanka.
  Fact 0.430: Rohini Marasinghe's term was 26 April 2013 to 2015.
  Fact 0.431: Rohini Marasinghe was appointed by Rajapaksa.
  Fact 0.432: Rohini Marasinghe studied at the University of Hartford.
  Fact 0.433: Rohini Marasinghe's prior position was Judge of the Court of Appeal.
  Fact 0.434: Buwaneka Aluwihare's term began 4 December 2013 and is ongoing.
  Fact 0.435: Buwaneka Aluwihare was appointed by Rajapaksa.
  Fact 0.436: Buwaneka Aluwihare studied at the University of London.
  Fact 0.437: Buwaneka Aluwihare's prior position was Additional Solicitor General of Sri Lanka.
  Fact 0.438: Sisira de Abrew was from the Western province.
  Fact 0.439: Sisira de Abrew's term began 7 May 2014 and is ongoing.
  Fact 0.440: Sisira de Abrew was appointed by Rajapaksa.
  Fact 0.441: Sisira de Abrew studied at Sri Lanka Law College.
  Fact 0.442: Sisira de Abrew's prior position was President of the Court of Appeal of Sri Lanka.
  Fact 0.443: Sarath de Abrew lived from 1953 to 2016.
  Fact 0.444: Sarath de Abrew's term began 7 May 2014 and ended in January 2016.
  Fact 0.445: Sarath de Abrew was appointed by Rajapaksa.
  Fact 0.446: Sarath de Abrew studied at Sri Lanka Law College.
  Fact 0.447: Sarath de Abrew's prior position was Judge of the Court of Appeal.
  Fact 0.448: Priyantha Jayawardena's term began 7 May 2014 and is ongoing.
  Fact 0.449: Priyantha Jayawardena was appointed by Rajapaksa.
  Fact 0.450: Priyantha Jayawardena studied at Sri Lanka Law College and the University of Aberdeen.
  Fact 0.451: Upali Abeyratne's term began 17 December 2014 and is ongoing.
  Fact 0.452: Upali Abeyratne was appointed by Rajapaksa.
  Fact 0.453: Upali Abeyratne's prior position was Judge of the Court of Appeal.
  Fact 0.454: Anil Goonaratne's term began 30 January 2015 and is ongoing.
  Fact 0.455: Anil Goonaratne was appointed by Sirisena.
  Fact 0.456: Anil Goonaratne studied at Sri Lanka Law College and the University of Exeter.
  Fact 0.457: Anil Goonaratne's prior position was Judge of the Court of Appeal.
  Fact 0.458: Kankanithanthri Chitrasiri's term began 3 December 2015 and is ongoing.
  Fact 0.459: Kankanithanthri Chitrasiri was appointed by Sirisena.
  Fact 0.460: Kankanithanthri Chitrasiri studied at the University of Sri Lanka and Queen Mary University of London.
  Fact 0.461: Kankanithanthri Chitrasiri's prior position was Judge of the Court of Appeal.
  Fact 0.462: Nalin Perera's term began 3 March 2016 and is ongoing.
  Fact 0.463: Nalin Perera was appointed by Sirisena.
  Fact 0.464: Nalin Perera's prior position was Judge of the Court of Appeal.
  Fact 0.465: Prasanna Sujeewa Jayawardena's term began 16 June 2016 and is ongoing.
  Fact 0.466: Prasanna Sujeewa Jayawardena was appointed by Sirisena.
  Fact 0.467: Prasanna Sujeewa Jayawardena studied at the University of Colombo.
  Fact 0.468: Prasanna Sujeewa Jayawardena's prior position was Judge of the Court of Appeal.

Retrieved items:
<item 1>
List of Justices of the Supreme Court of Sri Lanka
450 -- Rajapaksa Sri Lanka Law College President of the Court of Appeal of Sri Lanka Sarath de Abrew 1953 -- 2016 7 May 2014 -- January 2016 -- -- Rajapaksa Sri
Lanka Law College Judge of the Court of Appeal Priyantha Jayawardena † 7 May 2014 -- present -- 7003145000000000000 ♠ 1,450 -- Rajapaksa Sri Lanka Law College
University of Aberdeen Upali Abeyratne † 17 December 2014 -- present -- 7003122600000000000 ♠ 1,226 -- Rajapaksa Judge of the Court of Appeal Anil Goonaratne †
30 January 2015 -- present -- 7003118200000000000 ♠ 1,
</item 1>

<item 2>
List of Justices of the Supreme Court of Sri Lanka
Amerasinghe 1937 - 2013 -- -- -- A.R.N. Fernando -- -- -- Ameer Ismail 1990 -- 1994 -- -- Premadasa University of Ceylon Ceylon Law College President of the
Court of Appeal of Sri Lanka Andrew Ranjan Perera -- -- -- Pathmanathan Ramanathan 1932 - 2006 -- -- -- University of Wales President of the Court of Appeal of
Sri Lanka K.M.M.B. Kulatunge -- -- -- G.P.S.
</item 2>

<item 3>
List of Justices of the Supreme Court of Sri Lanka
Valentine Jayewardene 1877 -- 1927 1923 -- -- -- Manning Judge of the District Court of Colombo 56 Llwellyn Chisholm Dalton 14 March 1925 -- -- -- 57 Lewis
Matthew Maartensz 18 February 1935 -- -- -- 58 Maas Thajoon Akbar 1880 - 1944 -- -- -- University of Cambridge Solicitor General of Ceylon 59 Charles Ernest St.
John Branch 1865 - 1939 3 July 1925 -- 25 May 1926 3 July 1925 -- 25 May 1926 326 326 Durham University 60 Stanley Fisher 1867 - 11 December 1926 -- 1930 11
December 1926 -- 1930 Chief Justice of Trinidad and Tobago 61 Robert William Lyall Grant 17 February 1926 -- -- -- 62 Allen Drieberg 19 September 1927 -- -- --
63 Eugene Wilfred Jayewardene Western 1874 - 1932 May 1928 -- -- -- Fletcher Member of the Colombo Municipal Council 64 Philip James
</item 3>

<item 4>
List of Justices of the Supreme Court of Sri Lanka
Jump up ^ Current to most recent page update . Bibliography ( edit ) Amerasinghe , A.R.B. ( 1986 ) . The Supreme Court of Sri Lanka : the first 185 years .
Ratmalana , Sri Lanka : Sarvodaya Book Pub . Services . ISBN 955599000X . ( hide ) Supreme Court of Sri Lanka The Court Decisions Procedure History Court
Building Current membership Chief Justice Kanagasabapathy Sripavan Puisne Justices Priyasath Dep Eva Wansundera Buwaneka Aluwihare Sisira De Abrew Priyantha
Jayawardena Upaly Abeyratne Anil Goonaratne K.T.
</item 4>

<item 5>
List of Justices of the Supreme Court of Sri Lanka
Kulatunge -- -- -- G.P.S. de Silva -- 1991 -- 1999 Sarath N. Silva 1995 -- 16 September 1999 16 September 1999 -- 7 June 2009 3,552 Kumaratunga Sri Lanka Law
College University of Brussels President of the Court of Appeal of Sri Lanka Shirani Bandaranayake North Western 1958 - 30 October 1996 -- 13 January 2013 28
January 2015 -- 29 January 2015 18 May 2011 -- 13 January 2013 28 January 2015 -- 29 January 2015 5,919 + 1 606 + 1 Kumaratunga University of Colombo University
of London Lawyer C.V. Vigneswaran Western 1939 - March 2001 -- October 2004 -- -- Kumaratunga University of Ceylon Ceylon Law College Judge of the Court of
Appeal Asoka de Silva 1946 - 1 August 2001 -- 17 May 2011 8 June 2009 -- 17 May 2011 3,
</item 5>

<item 6>
List of Justices of the Supreme Court of Sri Lanka
Imam Central 9 January 2009 -- 20 February 2013 -- 1,503 -- Rajapaksa Judge of the Court of Appeal R.K.S. Suresh Chandra 17 June 2010 -- -- -- Rajapaksa Sathya
Hettige 10 July 2011 -- 18 April 2014 -- 1,013 -- Rajapaksa University of Colombo President of the Court of Appeal of Sri Lanka Priyasath Dep † 10 July 2011 --
present 02 March 2017 - present -- 7003248200000000000 ♠ 2,482 -- Rajapaksa University of Colombo International Institute of Social Studies Solicitor General of
Sri Lanka Eva Wansundera † North Western 7 July 2012 -- present -- 7003211900000000000 ♠ 2,
</item 6>

<item 7>
List of Justices of the Supreme Court of Sri Lanka
119 -- Rajapaksa Sri Lanka Law College University of Leicester Attorney General of Sri Lanka Mohan Peiris ♠ 15 January 2013 -- 28 January 2015 15 January 2013
-- 28 January 2015 743 743 Rajapaksa Sri Lanka Law College Attorney General of Sri Lanka Rohini Marasinghe 26 April 2013 -- 2015 -- -- Rajapaksa University of
Hartford Judge of the Court of Appeal Buwaneka Aluwihare † 4 December 2013 -- present -- 7003160400000000000 ♠ 1,604 -- Rajapaksa University of London
Additional Solicitor General of Sri Lanka Sisira de Abrew † Western 7 May 2014 -- present -- 7003145000000000000 ♠ 1,
</item 7>

<item 8>
List of Justices of the Supreme Court of Sri Lanka
Chitrasiri Nalin Perera All members List of all members by court by seat by time in office by education succession timeline List of Chief Justices acting by
time in office Other countries Law Portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_Justices_of_the_Supreme_Court_of_Sri_Lanka&oldid=806670013 '' Categories : Supreme Court of Sri Lanka
justices Lists of Sri Lankan judges Talk Contents About Wikipedia Add links This page was last edited on 23 October 2017 , at 13 : 22 . About Wikipedia
</item 8>

<item 9>
List of Justices of the Supreme Court of Sri Lanka
576 708 Kumaratunga University of Peradeniya University of Colombo University of Illinois at Urbana -- Champaign President of the Court of Appeal of Sri Lanka
Nihal Jayasinghe 19 February 2003 -- 29 February 2008 -- 1,836 -- Kumaratunga University of Colombo Judge of the Court of Appeal Nimal Dissanayake -- 23 April
2008 -- -- Kumaratunga Shiranee Tilakawardane 19 February 2003 - 5 April 2014 -- 1,901 -- Kumaratunga Sri Lanka Law College President of the Court of Appeal of
Sri Lanka Nimal Gamini Amaratunga January 2005 -- March 2014 -- -- Kumaratunga University of Colombo International Maritime Law Institute Judge of the Court of
Appeal Saleem Marsoof February 2005 -- 2014 -- -- Kumaratunga University of Colombo ,
</item 9>

<item 10>
List of Justices of the Supreme Court of Sri Lanka
239 + 1,346 1,346 Senanayake University of London Ceylon Law College Judge of the Court of Appeal 113 Christopher Weeramantry Western 1926 - 1967 -- 30 April
1972 -- -- Senanayake University of Ceylon University of London Commissioner of Assize 114 Oswald Leslie De Kretser III Western 1910 - 1968 -- 21 January 1972
-- -- Senanayake University College , Colombo University of London Judge of the District Court of Panadura 115 Samarapilimudalige Wijayatilake Central 1912 -
1990 1968 -- 16 May 1975 -- -- Senanayake University College ,
</item 10>
=== END ITEM id=f1fc59d38caa3305 ===

=== ITEM id=934a384b9b36202b ===
Gold sentences and facts:
Sentence 0: Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995
  Fact 0.0: Mumbai has the Marathi name मुंबई.
  Fact 0.1: Mumbai was previously named Bombay.
  Fact 0.2: Mumbai was renamed in 1995.

Retrieved items:
<item 1>
Renaming of cities in India
Realignment of the official Indian English name to an alternative local name ( edit ) Aside from changes to the official English spellings of local names there
have also been renaming proposals to realign the official name , hence the English name with an alternative local name . Ethnically sensitive examples include
the proposals by the Bharatiya Janata Party ( 1990 , 2001 ) to rename Ahmedabad to Karnavati and Allahabad to Prayag . These two proposals are changes from the
Muslim ( or Mughal ) name to a Hindu native name . These can be represented as a change from Urdu language to Hindi language , but since the two languages are
variants of Hindustani the proposal is effectively a cultural and ethno - religious proposal rather than a linguistic one . Adoption of name changes in India
and overseas ( edit ) Official name changes naturally take place quickly if not immediately in official government sources . Adoption may be slower among the
media in India and abroad , and among Indian authors . Important examples ( edit ) Main article : List of renamed places in India States ( edit ) Travancore -
Cochin → Kerala ( 1 November 1956 ) .
</item 1>

<item 2>
Renaming of cities in India
Jump up ^ Calcutta : A Cultural and Literary History - Page 3 Krishna Dutta - 2003 `` nationalist stance , like Bombay , which changed its name to Mumbai , or
Madras , which has become the unrecognisable Chennai , Calcutta has preferred a comparatively minor name change , which frankly is a bit of a multicultural
mishmash . '' Jump up ^ `` David Rumsey : Geographical Searching with MapRank Search ( beta ) '' . Rumsey.mapranksearch.com . Retrieved 2012 - 08 - 15 . Jump up
^ Beam , Christopher ( 2006 - 07 - 12 ) . `` How Bombay became Mumbai . - Slate Magazine '' . Slate.com . Retrieved 2012 - 08 - 15 .
</item 2>

<item 3>
Renaming of cities in India
The name changes are not without controversy among Indians . In several instances , the name change represents a struggle between a cosmopolitan elite and a
local , regional - language populace over defining the city in ways that go far beyond a simple change of name . '' Jump up ^ Steven I. Wilkinson Votes and
Violence : Electoral Competition and Ethnic Riots in India 2006 Page 23 `` The BJP proposed in 1990 and 2001 that Ahmedabad be renamed `` Karnavati . '' Hindu ,
June 1 1 , 2001 . Similar proposals have been made to rename Allahabad `` Prayag . '' Jump up ^ Cosmopolitanism - Page 73 Carol A. Breckenridge , Sheldon
Pollock , Homi K. Bhabha - 2002 `` In one sense , the decision to officialise the name Mumbai is part of a widespread Indian pattern of replacing names
associated with colonial rule with names associated with local , national , and regional heroes .
</item 3>

<item 4>
Renaming of cities in India
In the early years after Indian independence , many name changes were effected in northern India for English spellings of Hindi place names that had simply been
romanised inconsistently by the British administration -- such as the British spelling `` Jubbulpore , '' renamed `` Jabalpur '' ( जबलपुर ) among the first
changes in 1947 . These changes did not generate significant controversy . More recent and high - profile changes -- including renaming such major cities as
Calcutta to Kolkata -- have generated greater controversy . Since independence , such changes have typically been enacted officially by legislation at local or
national Indian government level , and may or may not then be adopted by the Indian media , particularly the influential Indian press . In the case of smaller
towns and districts which were less notable outside and inside India , and where a well known English name ( or exonym ) could not be said to exist , older
spellings used under British India may not have had any specific legislation other than changes in practice on the romanisation of indigenous Indian language
names .
</item 4>

<item 5>
Renaming of cities in India
from Trivandrum , respelled in 1991 Mumbai ( Marathi : मुंबई ) , from Bombay , renamed in 1995 Kochi ( Malayalam : കൊച്ചി ) , from Cochin , respelled in 1996
Chennai ( Tamil : சென்னை ) , from Madras , renamed in 1996 Kolkata ( Bengali : কলকাতা ) , from Calcutta , respelled in 2001 Kadapa ( Telugu : కడప ) , from
Cuddapah , respelled in 2005 Puducherry ( Tamil : புதுச்சேரி ) , from Pondicherry , renamed in 2006 Bengaluru ( Kannada : ಬೆಂಗಳೂರು ) , from Bangalore ,
</item 5>

<item 6>
Renaming of cities in India
Renaming of cities in India - Wikipedia Renaming of cities in India Jump to : navigation , search The renaming of cities in India started in 1947 following the
end of the British imperial period . Several changes were politically controversial , and not all proposed changes were implemented . Each had to be approved by
the central government in New Delhi . The renaming of states and territories in India has also taken place , but until recently with actual substantial name
changes in both local language and in English such as the old British state name of Travancore - Cochin to Kerala ( 1956 ) . The most notable recent exceptions
are Indian English spelling - changes of Orissa to Odisha ( March 2011 ) and the Union Territory of Pondicherry ( which includes the City of Pondicherry ) to
Puducherry .
</item 6>

<item 7>
Renaming of cities in India
Many people argue that such confusion can lead to indeterminate and / or unintended consequences See also ( edit ) India portal List of renamed Indian cities
and states Renaming in South Africa References ( edit ) Jump up ^ India and the World Bank : The Politics of Aid and Influence - Page 126 Jason A. Kirk - 2011
`` Orissa ( Note : This state was officially renamed Odisha in March 2011 ) '' Jump up ^ The Hindu Nationalist Movement and Indian Politics , 1925 to the 1990s
. - Page 134 Christophe Jaffrelot 1999 - `` The new state included Madhya Bharat , the Bhopal region , the former Vindhya Pradesh , Mahakoshal and Chhattisgarh
( the last two regions forming the Hindi - speaking parts in the former Madhya Pradesh ; see map , pp. xxii - xxiii ) . '' Jump up ^ Mira Kamdar Planet India :
How the Fastest Growing Democracy Is Transforming ... 2007 Author 's introduction Page xi `` India 's information - technology capital 's new name , should it
be adopted , will mean `` town of boiled beans . ''
</item 7>

<item 8>
Renaming of cities in India
For example , the change of Madras ( Tamil : மதராஸ் Madras ) to Chennai ( Tamil : சென்னை Chennai ) was reflected in many of India 's languages , and
incidentally in English , while the Tamil endonym had always been Chennai and remained unaffected by the change . Renaming in English ( edit ) Change in
official English spelling ( edit ) The renaming of cities is often specifically from English to Indian English in connection with that dialect 's internal
reforms . In other words , the city itself is not actually renamed in the local language , and the local name ( or endonym ) in the indigenous languages of
India does not change , but the official spelling in Indian English is amended . An example is the change from English `` Calcutta '' to English `` Kolkata ''
-- the local Bengali name ( কলকাতা Kôlkata ) did not change . Such changes in English spelling may be in order to better reflect a more accurate phonetic
transliteration of the local name , or may be for other reasons .
</item 8>

<item 9>
Renaming of cities in India
Madhya Bharat → Madhya Pradesh ( 1 November 1959 ) Madras State → Tamil Nadu ( 14 January 1969 ) Mysore State → Karnataka ( 1 November 1973 ) Uttaranchal →
Uttarakhand ( 1 January 2007 ) Orissa → Odisha ( 4 November 2011 ) Cities ( edit ) Notable city names that were officially changed by legislation after
independence include : Jabalpur ( Hindi : जबलपुर ) , from Jubbulpore , respelled in 1947 Jajmau ( Hindi : जाजमऊ ) , from Jajesmow , respelled in 1948 Kanpur (
Hindi : कानपुर ) , from Cawnpore , respelled in 1948 Vadodara ( Gujarati : વડોદરા ) , from Baroda , respelled in 1974 Thiruvananthapuram ( Malayalam :
തിരുവനന്തപുരം ) , from Trivandrum ,
</item 9>

<item 10>
Renaming of cities in India
It is an indigenizing toponymic '' Jump up ^ Reserve Bank of India 's instructions for banks & banking operations Reserve Bank of India 2001 Page 713 `` The new
name `` Mumbai '' should be reflected in both English and Hindi and the change in name is to be brought about in all official communications , name plates ,
sign boards , office seals , rubber stamps , etc . '' Jump up ^ Perveez Mody The Intimate State : Love - Marriage and the Law in Delhi Page 59 - 2008 ``
Throughout this book , I refer to India 's commercial capital as Bombay rather than Mumbai ... I am well aware of the name - change effected by an Act of the
Indian Parliament in 1997 that made the city officially ' Mumbai ' ... It is the same convention I adopt when referring to Calcutta rather than Kolkata . ''
Jump up ^ Pingali Sailaja Indian English Page 16 2009 `` Bombay is now called Mumbai , Madras is now Chennai and Calcutta is Kolkata , in an attempt to
de-anglicise them . In this work , the earlier names are retained since these names were used during the period that we mostly cover . ''
</item 10>
=== END ITEM id=934a384b9b36202b ===

=== ITEM id=f8bde41f7ddb2d1b ===
Gold sentences and facts:
Sentence 0: In the United States , there is an Animal Welfare Act that was produced in 1966 . This law protects animals in acts of research , transportation , and sale . Generally , animals are protected from any torture , neglect , or killing . There have been many amendments made towards this act to keep it updated . In order to be charged for animal abuse , there has to be prosecution and a trial . While there is only one act covering the entire United States , there are more current laws surrounding animal rights , which vary by state . Even though there are policies and severe consequences , the laws are not normally enforced with as much force or timeliness as regular crime .
  Fact 0.0: In the United States, there is an Animal Welfare Act.
  Fact 0.1: The Animal Welfare Act was produced in 1966.
  Fact 0.2: The Animal Welfare Act protects animals in acts of research, transportation, and sale.
  Fact 0.3: Under the Animal Welfare Act, animals are generally protected from any torture, neglect, or killing.
  Fact 0.4: Many amendments have been made to the Animal Welfare Act to keep it updated.
  Fact 0.5: To be charged for animal abuse, there has to be prosecution and a trial.
  Fact 0.6: Only one act covers the entire United States.
  Fact 0.7: There are more current laws surrounding animal rights, and these laws vary by state.
  Fact 0.8: There are policies and severe consequences for animal abuse.
  Fact 0.9: Animal abuse laws are not normally enforced with as much force or timeliness as regular crime.

Retrieved items:
<item 1>
Animal rights movement
ISSN 1187 - 7863 . doi : 10.1007 / s10806 - 015 - 9593 - 4 . ^ Jump up to : Wrenn , Corey Lee ( 2014 - 04 - 01 ) . `` Abolition Then and Now : Tactical
Comparisons Between the Human Rights Movement and the Modern Nonhuman Animal Rights Movement in the United States '' . Journal of Agricultural and Environmental
Ethics. 27 ( 2 ) : 177 -- 200 . ISSN 1187 - 7863 . doi : 10.1007 / s10806 - 013 - 9458 - 7 . ^ Jump up to : Evans , Erin ( 2010 - 07 - 01 ) . `` Constitutional
Inclusion of Animal Rights in Germany and Switzerland : How Did Animal Protection Become an Issue of National Importance ? '' . Society & Animals . 18 ( 3 ) :
231 -- 250 . ISSN 1568 - 5306 . doi : 10.1163 / 156853010X510762 . Jump up ^ `` Ethics : Animals . ''
</item 1>

<item 2>
Animal rights movement
An inclusive movement with strong group solidarity would decrease opportunity costs associated with participating ( e.g. , social stigmatization , lack of
alternatives , legal persecution ) and thus serve to increase and sustain participation in the movement . Current status of the movement ( edit ) The movement
is no longer viewed as hovering on the fringe . In the 1980s and 1990s , it was joined by a wide variety of academics and professionals , including lawyers ,
physicians , psychologists , veterinarians , and former vivisectionists , and is now a common subject of study in philosophy departments in Europe and North
America . Animal law courses are taught in 92 out of 180 law schools in the U.S. , and the movement has gained the support of senior legal scholars , including
Alan Dershowitz and Laurence Tribe of Harvard Law School . Chapters of animal rights law have been created in several state bar associations , and resolutions
related to animal rights are regularly proposed within the American Bar Association . In the 1980s , the movement became associated with punk subculture and
ideologies , particularly straight edge hardcore punk in the United States and anarcho - punk in the United Kingdom .
</item 2>

<item 3>
Animal rights movement
.. Movement Groups Animal Aid Animal Legal Defense Fund Animal Liberation Front British Union for the Abolition of Vivisection Centre for Animals and Social
Justice Chinese Animal Protection Network Direct Action Everywhere Farm Animal Rights Movement Great Ape Project Hunt Saboteurs Association In Defense of
Animals Korea Animal Rights Advocates Last Chance for Animals Mercy for Animals New England Anti-Vivisection Society Oxford Centre for Animal Ethics Oxford
Group People for the Ethical Treatment of Animals United Poultry Concerns more ... Parties AAEVP ( Canada ) Animal Justice Party ( Australia ) Animal Welfare
Party ( UK ) PACMA ( Spain ) Party for the Animals ( Netherlands ) Partito Animalista Italiano ( Italy ) Tierschutzpartei ( Germany ) V - Partei3 ( Germany )
People -- Animals -- Nature ( Portugal ) more ... Media Books Animals ' Rights : Considered in Relation to Social Progress ( 1894 ) Animals , Men and Morals (
1971 ) Animal Liberation ( 1975 ) The Case for Animal Rights ( 1983 ) The Lives of Animals ( 1999 ) Striking at the Roots ( 2008 ) An American Trilogy ( 2009 )
more ...
</item 3>

<item 4>
Animal rights movement
In the United States , there is an Animal Welfare Act that was produced in 1966 . This law protects animals in acts of research , transportation , and sale .
Generally , animals are protected from any torture , neglect , or killing . There have been many amendments made towards this act to keep it updated . In order
to be charged for animal abuse , there has to be prosecution and a trial . While there is only one act covering the entire United States , there are more
current laws surrounding animal rights , which vary by state . Even though there are policies and severe consequences , the laws are not normally enforced with
as much force or timeliness as regular crime . Strategy and Tactical repertoires ( edit ) Use of New information communication technologies ( ICTs ) ( edit )
New media , such as the Internet and email , have been used by Animal Rights Movement actors and countermovement actors in a variety of capacities . Radical
factions in the movement rely on websites , blogs , podcasts , videos , and online forums to engage in vegan outreach and other mobilization efforts and build
alliances , thus overcoming exclusion by dominant factions . The use of Internet has allowed the animal rights movement to spread transnationally .
</item 4>

<item 5>
Animal rights movement
, the Animal Enterprise Terrorism Act ) and does not economically or politically challenge extant systems . Smaller factions include groups focused around faith
- based animal rights theory and veganarchists , whose approach is characterized by a critique of capitalism on the grounds that it has led to mass nonhuman ,
human , and environmental exploitation . Such factionalizing , researchers have pointed out , is common to social movements and plays a role in sustaining their
health . History ( edit ) Further information : 20th century : Increase in animal use ; animal rights movement , Gary Francione , Tom Regan , Richard D. Ryder ,
and Peter Singer The Animal Rights Movement traces back to the animal protection movement in Victorian England , which was initiated by aristocratic moral
crusaders in response to the poor treatment of urban workhorses and stray dogs .
</item 5>

<item 6>
Animal rights movement
ISBN 1 - 56639 - 461 - 9 Harold D. Guither , Animal Rights : History and Scope of a Radical Social Movement ( Carbondale : Southern Illinois University Press ,
1998 ) . ISBN 0 - 8093 - 2199 - 8 James M. Jasper and Dorothy Nelkin , The Animal Rights Crusade : The Growth of a Moral Protest ( New York : The Free Press ,
1992 ) . ISBN 0 - 02 - 916195 - 9 ( hide ) Animal rights Topics Overview Abolitionism Animal protectionism Animal welfare Speciesism Vegaphobia Veganism more
...
</item 6>

<item 7>
Animal rights movement
Contents ( hide ) 1 Terms and factions 2 History 3 Gender , class , and other factors 4 Current status of the movement 5 Philosophical and legal aims 6 Legal
changes influenced by the movement 7 Strategy and tactical repertoires 7.1 Use of new information communication technologies ( ICTs ) 7.2 Undercover
surveillance 7.3 Boycott 7.4 Moral shocks 7.5 Direct action 7.6 Targeting researchers 8 Criminalization of direct action methods 9 Inter-movement activity 10
Countermovement 11 Notes 12 Further reading Terms and factions ( edit ) All animal liberationists believe that the individual interests of non-human animals
deserve recognition and protection , but the movement can be split into two broad camps . Animal rights advocates , or rights liberationists , believe that
these basic interests confer moral rights of some kind on the animals , and / or ought to confer legal rights on them ; see , for example , the work of Tom
Regan .
</item 7>

<item 8>
Animal rights movement
This association continues on into the 21st century , as evidenced by the prominence of vegan punk events such as Fluff Fest in Europe . Michael Socarras of
Greenberg Traurig told the Association of American Medical Colleges : `` There is a very important shift under way in the manner in which many people in law
schools and in the legal profession think about animals . This shift has not yet reached popular opinion . However , in ( the U.S. ) , social change has and can
occur through the courts , which in many instances do not operate as democratic institutions . Therefore , the evolution in elite legal opinion is extremely
significant ... '' Philosophical and legal aims ( edit ) Further information : Main animal rights philosophical approaches and Veganarchism The movement aims to
include animals in the moral community by putting the basic interests of non-human animals on an equal footing with the basic interests of human beings . A
basic interest would be , for example , not being made to suffer pain on behalf of other individual human or non-human animals . The aim is to remove animals
from the sphere of property and to award them personhood ; that is , to see them awarded legal rights to protect their basic interests .
</item 8>

<item 9>
Animal rights movement
It has been noted that the power of the animal rights movement in the United States is centralized in professionalized nonprofit organizations that prefer
moderated goals of welfare reform and tend to be isomorphic . The abolitionist faction diagnoses the problem as one of nonhuman animal use , offers a prognosis
including vegan outreach , and employ action frames including a rational and theoretic focus . Gary Francione , a leader in abolitionism , formed his approach
in response to the traditional movement 's focus on policy reform . Members of the abolitionist faction view policy reform as counterproductive and rely on
nonviolent education and moral suasion in their repertoire of contention . They see the promotion of veganism as a low - opportunity cost means of creating an
antispeciesist culture and posing an economic threat to the animal agriculture industrial complex . The direct action or militant faction includes in its
repertoire of contention property damage , open releases , intimidation , and direct violence . With these , they offer a prognosis to the problem of animal use
that focuses on changing society through force and fear . Animal rights actors often reject this faction , pointing to violence as a counterproductive tactic
that invites repression ( e.g.
</item 9>

<item 10>
Animal rights movement
Other early influences include : Upton Sinclair 's 1906 novel The Jungle , which drew attention to obscured slaughterhouse operations ; Henry Salt 's treatises
on nonhuman animal rights ; which drew from human abolitionist arguments for recognizing personhood of people considered to be property ; and the Alcott House
of New England , a community serving as a stop on the Underground Railroad and requiring its residents to eat a vegan diet . Philosopher Peter Singer The
contemporary movement is regarded as having been founded in the UK in the early 1970s by a group of Oxford university post-graduate philosophy students , now
known as the `` Oxford Group '' . The group was led by Rosalind and Stanley Godlovitch , graduate students of philosophy who had recently become vegetarians .
The Godlovitches met John Harris and David Wood , also philosophy graduates , who were soon persuaded of the arguments in favour of animal rights and themselves
became vegetarian . The group began to actively raise the issue with pre-eminent Oxford moral philosophers , including Professor Richard Hare , both personally
and in lectures . Their approach was based not on sentimentality ( `` kindness to dumb animals ' ) , but on the moral rights of animals .
</item 10>
=== END ITEM id=f8bde41f7ddb2d1b ===
